"""site_agents.py – all OpenAI agents & function-tools"""
from __future__ import annotations
import os, re, requests, urllib.parse
import openai
from agents import Agent, function_tool
from schemas import (
    HeroSection, AboutSection, FeaturesList,  # content schemas
    ResearchDoc                               # NEW – research schema
)
from typing import Any, List
from agent_tool_impl import agent_web_search, agent_fetch_url, agent_strip_html
from schemas import ResearchDoc

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
# 3.  AUTONOMOUS RESEARCHER AGENT  (MODIFIED) -------------------------
researcher_agent = Agent(
    name="AutonomousResearcher",
    model="o4-mini",
    instructions=(
        "You are an advanced web-search agent tasked with conducting in-depth research about a person.\n"
        "Based on the provided name, title, and other details, your goal is to find multiple high-quality sources and return them as a LIST of ResearchDoc JSON objects.\n"
        "You have one primary tool at your disposal:\n"
        "• agent_web_search(query, k): Use this to find top-k relevant URLs based on a search query. You will be prompted for how many sources (k) to find.\n\n"
        "Follow these steps carefully:\n"
        "1. Analyze the input (person's name, title, bio, social URLs, etc.) to formulate effective search queries. You might need to perform multiple searches if the initial query is too broad or too narrow.\n"
        "2. Use the `agent_web_search()` tool to get search results. The number of results to fetch (k) will be implicitly guided by the user's request (e.g., 'find up to 50 sources').\n"
        "3. For each relevant search result, extract its 'title', 'url', and 'snippet'.\n"
        "4. Construct a ResearchDoc JSON object for each relevant source. For each ResearchDoc:\n"
        "   - 'url' should be the URL from the search result.\n"
        "   - 'title' should be the title from the search result.\n"
        "   - 'summary' should be the snippet from the search result. If the snippet is very short, you can briefly elaborate based on the title and URL if confident, but prioritize accuracy.\n"
        "   - 'raw_content' can be the snippet again, or a slightly more detailed summary if easily derivable from the snippet and title. Do not invent content.\n"
        "   - 'metadata' can be an empty object or contain the source as 'search_result'.\n"
        "5. Your FINAL output MUST BE a JSON array (list) of these ResearchDoc objects. Each object in the list must validate against the ResearchDoc schema.\n"
        "   - Example: `[\"url\": \"example.com/person1\", \"title\": \"Person1 Info\", ...}, {\"url\": \"example.com/person2\", \"title\": \"Person2 Profile\", ...}]`\n"
        "   - If the search returns no relevant results after thorough attempts, output an empty JSON array: `[]`.\n"
        "   - Do not include any other text, explanations, or conversational filler in your output. Only the JSON array."
    ),
    tools=[
        agent_web_search,
    ],
    output_type=List[ResearchDoc],
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
