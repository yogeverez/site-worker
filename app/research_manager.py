# app/research_manager.py

import asyncio
import logging
import time
from typing import Any, Dict, List
from app.database import get_db
from app.process_status_tracker import ProcessStatusTracker
from app.agent_types import gen_trace_id, trace, Runner
from app.agents import user_data_collector_agent, researcher_agent, translator_agent, content_validator_agent
from app.agents.content_generation_orchestrator_agent import content_generation_orchestrator_agent
from app.user_data_collector import fetch_and_extract_social_data, summarize_user_data
from app.content_generation_agent import get_research_facts, save_content_section, save_content_section  # (imported for direct use if needed)

logger = logging.getLogger(__name__)

class ResearchManager:
    """
    Manages the multi-phase site generation workflow (user data collection, research, content generation, etc.).
    """
    def __init__(self, uid: str, session_id: int):
        self.uid = uid
        self.session_id = session_id
        self.db = get_db()
        self.status_tracker = ProcessStatusTracker(uid, session_id)
    
    async def run(self, user_input: Dict[str, Any], languages: List[str], mode: str = "full") -> Dict[str, Any]:
        """
        Execute the site generation workflow according to the specified mode.
        Returns a summary of results.
        """
        trace_id = gen_trace_id()
        with trace("SiteGenerationWorkflow", trace_id=trace_id):
            logger.info(f"[UID:{self.uid}] Starting site generation (mode={mode}, languages={languages})")
            self.status_tracker.update_phase("overall", "in_progress", metadata={"mode": mode})
            try:
                if mode == "full":
                    # FULL: run our content‐generation orchestrator (which handles data collection, research, then content)
                    primary_lang = languages[0] if languages else "en"
                    agent = content_generation_orchestrator_agent(self.uid, self.session_id, primary_lang)
                    runner = Runner()  # using SDK Runner
                    # Supply the user input as the initial user message (JSON format)
                    prompt = f"USER_INPUT_DATA = {user_input}"
                    result = await runner.run(agent, prompt, max_turns=40)
                    # After orchestrator completes, gather results:
                    # 1. User profile (if updated by user_data phase, otherwise use original input)
                    profile_dict = {}
                    if isinstance(user_input, dict):
                        profile_dict = user_input.copy()
                    # 2. Count research sources saved
                    sources_col = self.db.collection("research").document(self.uid).collection("sources")
                    sources_list = list(sources_col.stream())
                    source_count = len(sources_list)
                    # 3. Handle content for additional languages (translate from primary language content)
                    sections_generated: Dict[str, List[str]] = {}
                    # Primary language (first in list) content sections:
                    primary_lang = languages[0] if languages else "en"
                    content_col = self.db.collection("content").document(self.uid).collection(primary_lang)
                    primary_sections = [doc.id for doc in content_col.stream()]
                    sections_generated[primary_lang] = primary_sections
                    # Translate each section to other requested languages
                    for lang in languages[1:]:
                        sections_generated[lang] = []
                        for section in primary_sections:
                            # Fetch English content and translate it
                            eng_doc = content_col.document(section).get()
                            if eng_doc.exists:
                                eng_content = eng_doc.to_dict()
                                text_to_translate = __import__('json').dumps(eng_content, ensure_ascii=False)
                                # Use translator agent to translate JSON text
                                agent_t = translator_agent()
                                t_runner = Runner()
                                t_prompt = f"Translate the following {section} section content to {lang}:\n{text_to_translate}"
                                t_result = await t_runner.run(agent_t, t_prompt, max_turns=2)
                                translated_text = t_result.final_output if hasattr(t_result, 'final_output') else None
                                if translated_text:
                                    # Save translated content
                                    save_content_section(self.uid, section, translated_text, lang)
                                    sections_generated[lang].append(section)
                    # 4. Run content validation on all sections in primary language
                    for section in sections_generated.get(primary_lang, []):
                        sec_doc = content_col.document(section).get()
                        if sec_doc.exists:
                            sec_content = __import__('json').dumps(sec_doc.to_dict(), ensure_ascii=False)
                            validator = content_validator_agent(section)
                            v_runner = Runner()
                            v_prompt = f"Validate and correct the {section} section:\n{sec_content}"
                            v_result = await v_runner.run(validator, v_prompt, max_turns=2)
                            if v_result and hasattr(v_result, 'final_output'):
                                corrected = v_result.final_output
                                # Save corrections if any
                                save_content_section(self.uid, section, __import__('json').dumps(corrected, ensure_ascii=False), primary_lang)
                    # Mark overall process complete
                    self.status_tracker.complete_process()
                    logger.info(f"[UID:{self.uid}] ✅ Full generation completed: {source_count} sources, sections: {sections_generated}")
                    return {
                        "status": "success",
                        "user_profile": profile_dict,
                        "research_summary": {"sources_count": source_count},
                        "content_summary": {"sections_generated": sections_generated, "languages": languages},
                        "trace_id": trace_id
                    }
                
                elif mode == "research_only":
                    # Only perform user data collection and research phases, no content generation
                    self.status_tracker.update_phase("user_data_collection", "in_progress")
                    # Merge user input with any fetched social data (simulate using the same functions as agent)
                    social_results = []
                    for key, value in user_input.items():
                        if isinstance(value, str) and any(platform in value.lower() for platform in ["linkedin", "github", "twitter", "facebook", "instagram"]):
                            platform = next((p for p in ["linkedin", "github", "twitter", "facebook", "instagram"] if p in value.lower()), "unknown")
                            data = fetch_and_extract_social_data(value, platform)
                            social_results.append(data)
                    user_profile = summarize_user_data(user_input, social_results)
                    self.status_tracker.update_phase("user_data_collection", "completed")
                    # Research phase using researcher agent
                    self.status_tracker.update_phase("research", "in_progress")
                    agent = researcher_agent(self.uid, self.session_id)
                    runner = Runner()
                    # Compose a prompt for researcher agent (provide name and basic queries)
                    name = user_profile.name if hasattr(user_profile, 'name') else str(user_profile)
                    base_queries = [
                        f"{name} {user_input.get('current_title', '')}",
                        f"{name} {user_input.get('current_company', '')}",
                        f"{name} professional background"
                    ]
                    prompt = "ResearchQueries:\n" + "\n".join(base_queries)
                    result = await runner.run(agent, prompt, max_turns=20)
                    # Count saved sources
                    sources = list(self.db.collection("research").document(self.uid).collection("sources").stream())
                    source_count = len(sources)
                    self.status_tracker.update_phase("research", "completed", metadata={"sources_count": source_count})
                    self.status_tracker.complete_process()
                    logger.info(f"[UID:{self.uid}] ✅ Research-only completed: {source_count} sources saved.")
                    return {
                        "status": "success",
                        "user_profile": user_profile.dict() if hasattr(user_profile, 'dict') else user_profile,
                        "research_summary": {"sources_count": source_count},
                        "trace_id": trace_id
                    }
                
                elif mode == "content_only":
                    # Only content generation phase, assuming user_input already contains a complete profile.
                    self.status_tracker.update_phase("content_generation", "in_progress")
                    # We will proceed without research; use provided input directly for content generation.
                    sections_generated = {}
                    primary_lang = languages[0] if languages else "en"
                    # Invoke content generation orchestrator agent with an empty research context
                    agent = content_generation_orchestrator_agent(self.uid, self.session_id, primary_lang)
                    runner = Runner()
                    prompt = f"USER_PROFILE = {user_input}\n(No prior research available; generate content from profile alone.)"
                    await runner.run(agent, prompt, max_turns=15)
                    # Collect sections generated for primary language
                    content_col = self.db.collection("content").document(self.uid).collection(primary_lang)
                    sections_generated[primary_lang] = [doc.id for doc in content_col.stream()]
                    # (Optional) translator and validation can be applied similarly to the full mode if needed
                    self.status_tracker.update_phase("content_generation", "completed", metadata={"sections_generated": sections_generated.get(primary_lang, [])})
                    self.status_tracker.complete_process()
                    return {
                        "status": "success",
                        "user_profile": user_input,
                        "content_summary": {"sections_generated": sections_generated, "languages": languages},
                        "trace_id": trace_id
                    }
                
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
            
            except Exception as e:
                logger.error(f"[UID:{self.uid}] ❌ Error in workflow: {e}", exc_info=True)
                self.status_tracker.update_phase("overall", "failed", error=str(e))
                return {"status": "error", "error": str(e), "trace_id": trace_id}
