# app/research_manager.py

import asyncio
import logging
import time
from typing import Any, Dict, List

from agents import Runner
from agents.exceptions import MaxTurnsExceeded

from app.database import get_db
from app.process_status_tracker import ProcessStatusTracker, PhaseTimer
from app.agent_types import trace, gen_trace_id

from app.agents.user_data_collector_agent import user_data_collector_agent
from app.agents.researcher_agent import researcher_agent
from app.agents.content_generation_orchestrator_agent import content_generation_orchestrator_agent
from app.agents.translator_agent import translator_agent
from app.agents.content_validator_agent import content_validator_agent

from app.content_generation_agent import get_research_facts, save_content_section
from app.user_data_collector import fetch_and_extract_social_data, summarize_user_data

logger = logging.getLogger(__name__)


class ResearchManager:
    """
    Orchestrates the multi-phase workflow for site generation:
      1. User data collection
      2. Research
      3. Content generation
      4. Translation (optional)
      5. Validation
    """

    def __init__(self, uid: str, session_id: int):
        self.uid = uid
        self.session_id = session_id
        self.db = get_db()
        self.status = ProcessStatusTracker(uid, session_id)
        self.runner = Runner()

    async def run(self, user_input: Dict[str, Any], languages: List[str], mode: str = "full") -> Dict[str, Any]:
        trace_id = gen_trace_id()
        with trace("SiteGenerationWorkflow", trace_id=trace_id):
            logger.info(f"[UID:{self.uid}] ▶ Starting workflow (mode={mode}, langs={languages})")
            self.status.update_phase("overall", "in_progress", metadata={"mode": mode})

            try:
                if mode == "full":
                    primary = languages[0] if languages else "en"

                    profile = await self._phase_user_data(user_input)
                    await self._phase_research(profile)
                    facts_json = self._fetch_research_facts()
                    sections = await self._phase_content_generation(profile, facts_json, primary)
                    await self._phase_translation(sections, primary, languages[1:])
                    await self._phase_validation(primary)

                    self.status.complete_process()
                    logger.info(f"[UID:{self.uid}] ✅ Full run complete – sections: {sections}")
                    return {
                        "status": "success",
                        "user_profile": profile,
                        "sections": sections,
                        "trace_id": trace_id,
                    }

                elif mode == "research_only":
                    profile = await self._phase_user_data(user_input, collect_social=True)
                    count = await self._phase_research(profile)
                    self.status.complete_process()
                    return {
                        "status": "success",
                        "user_profile": profile,
                        "research_summary": {"sources_count": count},
                        "trace_id": trace_id,
                    }

                elif mode == "content_only":
                    primary = languages[0] if languages else "en"
                    sections = await self._phase_content_generation(user_input, None, primary)
                    self.status.complete_process()
                    return {
                        "status": "success",
                        "user_profile": user_input,
                        "content_summary": {"sections_generated": sections, "languages": languages},
                        "trace_id": trace_id,
                    }

                else:
                    raise ValueError(f"Unsupported mode: {mode}")

            except Exception as e:
                logger.exception(f"[UID:{self.uid}] ❌ Error in workflow")
                self.status.update_phase("overall", "failed", error=str(e))
                return {"status": "error", "error": str(e), "trace_id": trace_id}

    async def _phase_user_data(self, user_input: Dict[str, Any], collect_social: bool = False) -> Dict[str, Any]:
        """Run the User Data Collector agent (or ad-hoc social merge)."""
        with PhaseTimer(self.status, "user_data_collection"):
            if collect_social:
                # merge social links manually
                social_results = []
                for k, v in user_input.items():
                    if isinstance(v, str) and any(p in v.lower() for p in ["linkedin", "github", "twitter", "facebook", "instagram"]):
                        platform = next((p for p in ["linkedin","github","twitter","facebook","instagram"] if p in v.lower()), "unknown")
                        social_results.append(fetch_and_extract_social_data(v, platform))
                profile = summarize_user_data(user_input, social_results)
                logger.info(f"[UID:{self.uid}] ✔ Merged social data into profile")
                # Ensure the profile is serializable
                if hasattr(profile, 'dict'):
                    return profile.dict()
                elif hasattr(profile, 'model_dump'):
                    return profile.model_dump()
                return profile

            # use the Agent
            agent = user_data_collector_agent()
            prompt = f"USER_INPUT_DATA = {user_input}"
            result = await self.runner.run(agent, prompt, max_turns=20)
            profile = result.final_output if hasattr(result, "final_output") else result
            logger.info(f"[UID:{self.uid}] ✔ Agent-produced profile")
            # Ensure the profile is serializable
            if hasattr(profile, 'dict'):
                return profile.dict()
            elif hasattr(profile, 'model_dump'):
                return profile.model_dump()
            return profile

    async def _phase_research(self, profile: Dict[str, Any]) -> int:
        """Run the Researcher agent to populate Firestore with sources."""
        with PhaseTimer(self.status, "research"):
            agent = researcher_agent(self.uid, self.session_id)
            prompt = f"USER_PROFILE = {profile}"
            try:
                await self.runner.run(agent, prompt, max_turns=40)
            except MaxTurnsExceeded:
                logger.warning(f"[UID:{self.uid}] ⚠ Research turns exceeded; proceeding")
            # count documents
            sources = list(self.db.collection("research").document(self.uid).collection("sources").stream())
            count = len(sources)
            logger.info(f"[UID:{self.uid}] ✔ Research saved {count} sources")
            return count

    def _fetch_research_facts(self) -> str:
        """Retrieve up to 5 snippets from research for seeding content."""
        with PhaseTimer(self.status, "content_generation"):
            facts = get_research_facts(self.uid, self.session_id)
            logger.debug(f"[UID:{self.uid}] Research facts: {facts}")
            return facts

    async def _phase_content_generation(self, profile: Dict[str, Any], facts_json: str, lang: str) -> List[str]:
        """Run ContentGenerationOrchestrator and return the list of generated section IDs."""
        with PhaseTimer(self.status, "content_generation"):
            agent = content_generation_orchestrator_agent(self.uid, self.session_id, lang)
            prompt = f"USER_PROFILE = {profile}"
            if facts_json:
                prompt += f"\nRESEARCH_FACTS = {facts_json}"
            await self.runner.run(agent, prompt, max_turns=60)
            col = self.db.collection("content").document(self.uid).collection(lang)
            sections = [doc.id for doc in col.stream()]
            logger.info(f"[UID:{self.uid}] ✔ Generated sections: {sections}")
            return sections

    async def _phase_translation(self, sections: List[str], src_lang: str, tgt_langs: List[str]):
        """Translate each section from src_lang into each language in tgt_langs."""
        if not tgt_langs:
            return
        with PhaseTimer(self.status, "translation"):
            for lang in tgt_langs:
                logger.info(f"[UID:{self.uid}] ─ Translating to '{lang}'")
                for sec in sections:
                    doc = self.db.collection("content").document(self.uid).collection(src_lang).document(sec).get()
                    if not doc.exists:
                        continue
                    text = str(doc.to_dict())
                    agent = translator_agent()
                    try:
                        result = await self.runner.run(agent, f"Translate to {lang}:\n{text}", max_turns=10)
                    except MaxTurnsExceeded:
                        logger.warning(f"[UID:{self.uid}] ⚠ Translation turns exceeded for '{sec}'->{lang}")
                        continue
                    translated = result.final_output if hasattr(result, "final_output") else result
                    save_content_section(self.uid, sec, translated, lang)
                    logger.info(f"[UID:{self.uid}] ✔ Saved translation '{sec}'->{lang}")

    async def _phase_validation(self, lang: str):
        """Validate each generated section in the given language."""
        with PhaseTimer(self.status, "content_generation"):
            col = self.db.collection("content").document(self.uid).collection(lang)
            for sec in [doc.id for doc in col.stream()]:
                logger.info(f"[UID:{self.uid}] ─ Validating '{sec}'")
                doc = col.document(sec).get()
                content = str(doc.to_dict())
                agent = content_validator_agent(sec)
                try:
                    result = await self.runner.run(agent, f"Validate:\n{content}", max_turns=10)
                except MaxTurnsExceeded:
                    logger.warning(f"[UID:{self.uid}] ⚠ Validation turns exceeded for '{sec}'")
                    continue
                corrected = result.final_output if hasattr(result, "final_output") else result
                save_content_section(self.uid, sec, corrected, lang)
                logger.info(f"[UID:{self.uid}] ✔ Validation saved for '{sec}'")
