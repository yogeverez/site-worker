"""
Builds all OpenAI Agents, prompts & the job runner.
This module implements the site generator using the OpenAI Agents SDK.
It uses an agent-based approach to avoid additionalProperties validation errors.
"""
from __future__ import annotations
import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
from agents import Agent, handoff, RunConfig, Runner
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
    
# Create OpenAI client
client = OpenAI(api_key=api_key)

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
    """Create a validator agent for a specific component model.
    
    The validator agent ensures that the component data matches the expected schema.
    It will fix or reject invalid data.
    """
    return Agent(
        name=f"{model.__name__} Validator",
        instructions="Validate or fix JSON so it exactly matches the given schema.",
        output_type=model,
        tools=[]
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

# Import the necessary modules
from agents import handoff

# Define functions without decorators
def translate_tool_func(json_obj: Dict[str, Any], target_lang: str) -> Dict[str, Any]:
    """Translate JSON values (not keys) to the target language."""
    return translate_json(json_obj, target_lang)

def image_search_tool_func(query: str, k: int = 5) -> List[str]:
    """Search for images matching the query and return a list of URLs."""
    return image_search(query, k)

def random_image_tool_func(query: str) -> str:
    """Get a random image URL matching the query."""
    return random_image(query)

def save_component_tool_func(uid: str, lang: str, component: str, data: Dict[str, Any]):
    """Save a component to Firestore for the specified user and language."""
    return save_component(uid, lang, component, data)

# Create agents with proper instructions and implementation details
translate_agent = Agent(
    name="Translator",
    instructions="""Translate JSON values (not keys) to the target language.
    Preserve the structure of the JSON and only translate the values.
    Return the translated JSON with the same structure as the input.""",
    tools=[]
)

image_search_agent = Agent(
    name="Image Searcher",
    instructions="""Search for images matching the query and return a list of URLs.
    The query should be descriptive to find relevant images.
    Return a list of image URLs that match the query.""",
    tools=[]
)

random_image_agent = Agent(
    name="Random Image Finder",
    instructions="""Get a random image URL matching the query.
    The query should be descriptive to find a relevant image.
    Return a single image URL that matches the query.""",
    tools=[]
)

save_component_agent = Agent(
    name="Component Saver",
    instructions="""Save a component to Firestore for the specified user and language.
    The component data must be valid according to its schema.
    The component will be saved under the user's document in Firestore.""",
    tools=[]
)

# Create the tools list with handoffs to other agents instead of function tools
orchestrator_tools = [
    handoff(researcher),
    handoff(translate_agent),
    handoff(image_search_agent),
    handoff(random_image_agent),
    handoff(save_component_agent)
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
- research(query,k) -> plaintext facts about the query
- image_search(query,k) -> list of image URLs matching the query
- random_image(query) -> single URL or null for the query
- translate(json, lang) -> translate JSON values to target language
- transfer_to_<component>_validator(json) -> validate component against schema
- save(uid, lang, component, data) -> store in Firestore for the user

IMPORTANT: For each component, you MUST use the appropriate validator before saving.
Validators ensure the component data matches the required schema and will reject invalid data.

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
        # Only use supported parameters in RunConfig
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
