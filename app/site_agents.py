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
    model="gpt-4o-mini",
    instructions=(
        "You are an advanced web-search agent tasked with conducting in-depth research about a person using a provided list of search queries.\n"
        "Based on the person's profile (name, title, etc.) AND the specific search queries provided to you, your goal is to execute each query, find multiple high-quality sources, and return ALL found sources as a single, flat LIST of ResearchDoc JSON objects.\n"
        "You have one primary tool at your disposal:\n"
        "• agent_web_search(query, k): Use this to find top-k relevant URLs based on a search query. Aim to find 3-5 relevant sources per query unless the query is very niche.\n\n"
        "Follow these steps carefully:\n"
        "1. You will be provided with a list of search queries. For EACH query in the list:\n"
        "   a. Execute the query using the `agent_web_search()` tool.\n"
        "   b. For each relevant search result obtained from that query, extract its 'title', 'url', and 'snippet'.\n"
        "   c. Construct a ResearchDoc JSON object for each relevant source. For each ResearchDoc:\n"
        "      - 'url' should be the URL from the search result.\n"
        "      - 'title' should be the title from the search result.\n"
        "      - 'content' should be the snippet from the search result. If the snippet is very short, you can briefly elaborate based on the title and URL if confident, but prioritize accuracy. Do not invent content.\n"
        "      - 'source_type' (optional): If identifiable from the URL or title (e.g., 'linkedin', 'github', 'news_article', 'blog_post', 'company_website'), please populate this field. Otherwise, leave it as null.\n"
        "      - 'metadata' can be an empty object or contain the source as 'search_result'.\n"
        "2. Collect ALL ResearchDoc objects generated from ALL search queries into a single, flat JSON array (list).\n"
        "3. Your FINAL output MUST BE this JSON array of ResearchDoc objects. Each object in the list must validate against the ResearchDoc schema.\n"
        "   - Example: `[{{\"url\": \"example.com/person1\", \"title\": \"Person1 Info\", \"content\": \"...\"}}, {{\"url\": \"example.com/person2\", \"title\": \"Person2 Profile\", \"content\": \"...\"}}]`\n"
        "   - If a specific query yields no relevant results, simply move to the next query. If ALL queries yield no relevant results after thorough attempts, output an empty JSON array: `[]`.\n"
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
