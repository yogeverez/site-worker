"""
Builds all OpenAI Agents, prompts & the job runner.
"""
from __future__ import annotations
import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
from agents import Agent, handoff
from google.cloud import firestore
from tools import (
    fetch_url, strip_html, web_search, image_search, random_image, save_component
)
import schemas as sc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY environment variable is not set")
    
# Create OpenAI client with default model set to gpt-4o-mini
client = OpenAI(api_key=api_key, default_model="gpt-4o-mini")

# ------------------------------------------------------------------ #
# 1.  Sub-agents
# ------------------------------------------------------------------ #
researcher = Agent(
    name="Researcher",
    instructions=(
        "You are a diligent researcher.  "
        "Use the tools below to collect concise facts:\n"
        "• search(query,k)\n• fetch(url) -> html\n• strip_html(html)\n\n"
        "Return PLAINTEXT bullet points relevant to the person/company only."
    ),
    tools=[web_search, fetch_url, strip_html]
)

def make_validator(model):
    return Agent(
        name=f"{model.__name__} Validator",
        instructions="Validate or fix JSON so it exactly matches the given schema.",
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
from agents import function_tool

# Wrap the function tools
@function_tool
def translate_tool(json_obj: Dict[str, Any], target_lang: str) -> Dict[str, Any]:
    return translate_json(json_obj, target_lang)

@function_tool
def image_search_tool(query: str, k: int = 5) -> List[str]:
    return image_search(query, k)

@function_tool
def random_image_tool(query: str) -> str:
    return random_image(query)

@function_tool
def save_component_tool(uid: str, lang: str, component: str, data: Dict[str, Any]):
    return save_component(uid, lang, component, data)

# Create the tools list
orchestrator_tools = [
    handoff(researcher),
    translate_tool,
    image_search_tool,
    random_image_tool,
    save_component_tool
]

# Add validators to tools list
for comp, agent in validators.items():
    orchestrator_tools.append(handoff(agent))

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
    name="SiteGenerator Orchestrator",
    instructions=orchestrator_prompt,
    tools=orchestrator_tools
)

# ------------------------------------------------------------------ #
# 3.  Job runner
# ------------------------------------------------------------------ #
def run_generation_job(uid: str, languages: List[str]):
    import asyncio
    from agents import Runner
    
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

    async def run_for_language(lang):
        prompt = f"Generate website content for user {uid} in language {lang}"
        context = {
            "uid": uid,
            "lang": lang,
            "seed": base_seed
        }
        
        # Create a run configuration with the gpt-4o-mini model
        from agents import RunConfig
        run_config = RunConfig(model="gpt-4o-mini")
        
        # Run the agent with the configuration
        result = await Runner.run(
            orchestrator,
            input=prompt,
            context=context,
            max_turns=60,
            run_config=run_config
        )
        print(f"[{uid}] Generation finished for lang={lang}")
        return result

    # Run the async function in a synchronous context
    for lang in languages:
        asyncio.run(run_for_language(lang))
