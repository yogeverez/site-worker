"""
Builds all OpenAI Agents, prompts & the job runner.
"""
from __future__ import annotations
from typing import List, Dict, Any
from openai import OpenAI
from agents import Agent, handoff
from google.cloud import firestore
from tools import (
    fetch_url, strip_html, web_search, image_search, random_image, save_component
)
import schemas as sc

client = OpenAI()

# ------------------------------------------------------------------ #
# 1.  Sub-agents
# ------------------------------------------------------------------ #
researcher = Agent(
    system=(
        "You are a diligent researcher.  "
        "Use the tools below to collect concise facts:\n"
        "• search(query,k)\n• fetch(url) -> html\n• strip_html(html)\n\n"
        "Return PLAINTEXT bullet points relevant to the person/company only."
    ),
    tools={
        "search": web_search,
        "fetch": fetch_url,
        "strip_html": strip_html,
    },
)

def make_validator(model):
    return Agent(
        system="Validate or fix JSON so it exactly matches the given schema.",
        output_type=model
    )

validators: Dict[str, Agent] = {
    "hero": make_validator(sc.HeroComponent),
    "about": make_validator(sc.AboutComponent),
    "featuresList": make_validator(sc.FeaturesListComponent),
    "portfolioGrid": make_validator(sc.PortfolioGridComponent),
    "experienceTimeline": make_validator(sc.ExperienceTimelineComponent),
    "testimonials": make_validator(sc.TestimonialsComponent),
    "contact": make_validator(sc.ContactComponent),
    "newsletterSignup": make_validator(sc.NewsletterSignupComponent),
    "schedule": make_validator(sc.ScheduleComponent),
}

# Simple translator tool (function, not agent) -----------------------
def translate_json(json_obj: Dict[str, Any], target_lang: str) -> Dict[str, Any]:
    """Use GPT-4o to translate JSON **values** into target language."""
    content = (
        "Translate the JSON values (NOT keys) to {lang}. "
        "Return valid JSON only.  Do not add or delete keys.\n\n"
        "JSON:\n```json\n{json}\n```"
    ).format(lang=target_lang, json=json_obj)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content  # already JSON string

# ------------------------------------------------------------------ #
# 2.  Orchestrator agent
# ------------------------------------------------------------------ #
orchestrator_tools = {
    "research": handoff(researcher),
    "translate": translate_json,
    "image_search": image_search,
    "random_image": random_image,
    "save": save_component,
}
# add validators to tools dict
for comp, agent in validators.items():
    orchestrator_tools[f"validate_{comp}"] = handoff(agent)

orchestrator_prompt = """
You are the SiteGenerator Orchestrator.

For user "{uid}" and language "{lang}" create COMPLETE JSON for components:
hero, about, featuresList, portfolioGrid, experienceTimeline,
testimonials, contact, newsletterSignup, schedule.

TOOLS:
- research(query,k) -> plaintext facts
- image_search(query,k) -> list of image URLs
- random_image(query) -> single URL or null
- translate(json, lang) -> translate JSON values
- validate_<component>(json) -> fix/validate to schema
- save(component, json) -> store in Firestore (uid, lang)

FLOW:
1. research(name + job title + social links) to gather facts.
2. Draft each component JSON (English).
3. For hero choose backgroundImageUrl via random_image or leave null.
4. If lang != "en", translate(draft, lang).
5. validate_<component>() until it passes.
6. save().
End with DONE.
"""

orchestrator = Agent(
    system=orchestrator_prompt,
    tools=orchestrator_tools,
)

# ------------------------------------------------------------------ #
# 3.  Job runner
# ------------------------------------------------------------------ #
def run_generation_job(uid: str, languages: List[str]):
    fs = firestore.Client()
    wizard_ref = (
        fs.collection("users").document(uid)
        .collection("siteInputDocument").document("siteInputDocument")
    )
    wizard = wizard_ref.get().to_dict()
    if not wizard:
        print(f"[{uid}] No wizard data, skipping.")
        return

    # Seed for the agent
    base_seed = {k: v for k, v in wizard.items() if k not in ("createdAt", "updatedAt")}

    for lang in languages:
        prompt = orchestrator_prompt.format(uid=uid, lang=lang)
        orchestrator.run(
            prompt,
            extra_inputs={
                "uid": uid,
                "lang": lang,
                "seed": base_seed,
            },
            model="gpt-4o-mini",
            max_calls=60,
        )
        print(f"[{uid}] Generation finished for lang={lang}")
