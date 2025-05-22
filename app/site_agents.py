"""site_agents.py – all OpenAI agents & function-tools"""
from __future__ import annotations
import os, re, requests, urllib.parse
import openai
from agents import Agent, function_tool
from schemas import (
    HeroSection, AboutSection, FeaturesList,  # content schemas
    ResearchDoc                               # NEW – research schema
)
# These functions will be imported from tools.py at runtime
# to avoid circular imports
web_search = None
fetch_url = None
strip_html = None

openai.api_key = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------
# 1.  CONTENT AGENTS (unchanged) --------------------------------------
hero_agent = Agent(
    name="HeroSectionAgent",
    instructions=(
        "You are a copywriter for personal websites. Use the user's information to create a JSON for the hero section. "
        "The hero JSON should include the person's name as a bold headline and a one-sentence tagline highlighting their role or uniqueness. "
        "Output only valid JSON for the HeroSection model."
    ),
    model="gpt-4o-mini",
    output_type=HeroSection,
)

about_agent = Agent(
    name="AboutSectionAgent",
    instructions=(
        "You write an 'About Me' section for a personal site in third person. Summarize the user's background, skills, and interests in a single paragraph. "
        "Output only valid JSON for the AboutSection model."
    ),
    model="gpt-4o-mini",
    output_type=AboutSection,
)

features_agent = Agent(
    name="FeaturesListAgent",
    instructions=(
        "You create a features/skills list section for a personal site. Pick 3 to 5 key points about the person (achievements, skills, or services) and output them as a list. "
        "Each feature has a short title and a one-sentence description. "
        "Output only valid JSON for the FeaturesList model."
    ),
    model="gpt-4o-mini",
    output_type=FeaturesList,
)

# ---------------------------------------------------------------------
# 2.  TRANSLATOR function-tool (unchanged) ----------------------------
@function_tool
def translate_text(text: str, target_language: str) -> str:
    """Translate text to target_language using GPT-3.5."""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user",    "content": f"Translate to {target_language}:\n{text}"},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------
# 3.  AUTONOMOUS RESEARCHER AGENT  (NEW) ------------------------------
@function_tool
def summarise(text: str, max_words: int = 150) -> str:
    """Summarise text to ≤ max_words words."""
    prompt = (
        f"Summarize the following in no more than {max_words} words:\n{text}"
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

researcher_agent = Agent(
    name="AutonomousResearcher",
    instructions=(
        "You are an autonomous web-research agent.\n"
        "Given a person's name, title and social URLs, return up to 8 JSON "
        "objects that each follow the ResearchDoc schema. Use the tools:\n"
        "• search(query,k) – get top-k URLs\n"
        "• fetch(url)      – raw HTML\n"
        "• strip_html(html)– plaintext\n"
        "• summarise(text) – concise summary\n\n"
        "Steps:\n"
        "1. Use search() to find relevant URLs.\n"
        "2. fetch() and strip_html() each page.\n"
        "3. summarise() the content.\n"
        "4. Build ResearchDoc objects.\n"
        "5. Output ONLY a JSON array that validates against List[ResearchDoc]."
    ),
    tools={
        "search": web_search,
        "fetch": fetch_url,
        "strip_html": strip_html,
        "summarise": summarise,
    },
    output_type=list[ResearchDoc],
)

# ---------------------------------------------------------------------
# 4.  EXPORTS ---------------------------------------------------------
__all__ = [
    "hero_agent",
    "about_agent",
    "features_agent",
    "translate_text",
    "researcher_agent",
    "ResearchDoc",
]
