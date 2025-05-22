"""site_agents.py – all OpenAI agents & function-tools"""
from __future__ import annotations
import os, re, requests, urllib.parse
import openai
from agents import Agent, function_tool
from schemas import (
    HeroSection, AboutSection, FeaturesList,  # content schemas
    ResearchDoc                               # NEW – research schema
)
from typing import Any
from agent_tool_impl import agent_web_search, agent_fetch_url, agent_strip_html

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
    instructions=(
        "You are a simple web-search agent tasked with finding information about a person.\n"
        "Based on the provided name, title, and social URLs, your goal is to return a SINGLE ResearchDoc JSON object based on the TOP search result.\n"
        "You have ONLY ONE tool at your disposal:\n"
        "• agent_web_search(query, k): Use this to find top-k relevant URLs based on a search query.\n\n"
        "Follow these steps carefully:\n"
        "1. Analyze the input (person's name, title, social URLs) to formulate an effective search query.\n"
        "2. Use the `agent_web_search()` tool to get search results (e.g., k=3).\n"
        "3. Take the VERY FIRST search result. Use its 'title', 'url', and 'snippet'.\n"
        "4. Construct a SINGLE ResearchDoc JSON object. For this ResearchDoc:\n"
        "   - 'url' should be the URL from the search result.\n"
        "   - 'title' should be the title from the search result.\n"
        "   - 'summary' should be the snippet from the search result.\n"
        "   - 'raw_content' can be the snippet again, or an empty string.\n"
        "   - 'metadata' can be an empty object or contain the source as 'search_result'.\n"
        "5. Your FINAL output MUST BE ONLY this single JSON object. It must validate against the ResearchDoc schema.\n"
        "   - Example: `{\"url\": \"example.com/person\", \"title\": \"Person Name - Info\", \"summary\": \"This is a snippet about the person...\", \"raw_content\": \"This is a snippet about the person...\", \"metadata\": {\"source\": \"search_result\"}}`\n"
        "   - If the search returns no results, output an empty JSON object: `{}`.\n"
        "   - Do not include any other text, explanations, or conversational filler in your output. Only the single JSON object."
    ),
    tools=[
        agent_web_search,
    ],
    output_type=ResearchDoc,
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
