"""site_agents.py – all OpenAI agents & function-tools"""
from __future__ import annotations
import os, re, requests, urllib.parse
import openai
from openai import RateLimitError, APIStatusError # Import for error handling
from agents import Agent, function_tool
from schemas import (
    HeroSection, AboutSection, FeaturesList,  # content schemas
    ResearchDoc                               # NEW – research schema
)
from typing import Any, List
from agent_tool_impl import agent_web_search, agent_fetch_url, agent_strip_html
from schemas import ResearchDoc

# Remove global API key initialization: openai.api_key = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------
# 1.  CONTENT AGENTS --------------------------------------------------
def get_hero_agent(client: openai.OpenAI) -> Agent:
    hero_instructions = (
        "You are a copywriter for personal websites. Use the user's information to create a JSON for the hero section. "
        "The hero JSON should include the person's name as a bold headline and a one-sentence tagline highlighting their role or uniqueness. "
        "Output only valid JSON for the HeroSection model."
    )
    return Agent(
        name="HeroSectionAgent",
        instructions=hero_instructions,
        model="gpt-4o-mini",
        output_type=HeroSection,
        llm=client # Pass the configured client
    )

def get_about_agent(client: openai.OpenAI) -> Agent:
    about_instructions = (
        "You write an 'About Me' section for a personal site in third person. Summarize the user's background, skills, and interests in a single paragraph. "
        "Output only valid JSON for the AboutSection model."
    )
    return Agent(
        name="AboutSectionAgent",
        instructions=about_instructions,
        model="gpt-4o-mini",
        output_type=AboutSection,
        llm=client # Pass the configured client
    )

def get_features_agent(client: openai.OpenAI) -> Agent:
    features_instructions = (
        "You create a features/skills list section for a personal site. Pick 3 to 5 key points about the person (achievements, skills, or services) and output them as a list. "
        "Each feature has a short title and a one-sentence description. "
        "Output only valid JSON for the FeaturesList model."
    )
    return Agent(
        name="FeaturesListAgent",
        instructions=features_instructions,
        model="gpt-4o-mini",
        output_type=FeaturesList,
        llm=client # Pass the configured client
    )

# ---------------------------------------------------------------------
# 2.  TRANSLATOR function-tool ----------------------------------------
@function_tool
async def translate_text(client: openai.OpenAI, text: str, target_language: str) -> str:
    """Translate text to target_language using an OpenAI model."""
    try:
        resp = await client.chat.completions.create(
            model="gpt-3.5-turbo-0125", # Consider gpt-4o-mini for consistency/cost
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user",    "content": f"Translate to {target_language}:\n{text}"},
            ],
            temperature=0.3,
        )
        translation = resp.choices[0].message.content.strip()
        if not translation:
            # Fallback or error logging if translation is empty
            print(f"Warning: Translation for '{text[:30]}...' to {target_language} resulted in empty string.")
            return text # Return original text as a fallback
        return translation
    except RateLimitError as e:
        print(f"OpenAI API rate limit exceeded during translation: {e}")
        return f"Error: Translation failed due to rate limit. Original: {text}" # Indicate error
    except APIStatusError as e:
        print(f"OpenAI API status error during translation: {e}")
        return f"Error: Translation failed due to API error. Original: {text}" # Indicate error
    except Exception as e:
        print(f"An unexpected error occurred during translation: {e}")
        return f"Error: Translation failed. Original: {text}" # Indicate error

# ---------------------------------------------------------------------
# 3.  AUTONOMOUS RESEARCHER AGENT -------------------------------------
def get_researcher_agent(client: openai.OpenAI) -> Agent:
    researcher_instructions = """You are an advanced web-search agent tasked with conducting in-depth research about a person using a provided list of search queries.
Based on the person's profile (name, title, etc.) AND the specific search queries provided to you, your goal is to execute each query, find multiple high-quality sources, and return ALL found sources as a single, flat LIST of ResearchDoc JSON objects.
You have one primary tool at your disposal:
• agent_web_search(query, k): Use this to find top-k relevant URLs based on a search query. Aim to find 3-5 relevant sources per query unless the query is very niche.

Follow these steps carefully:
1. You will be provided with a list of search queries. For EACH query in the list:
   a. Execute the query using the `agent_web_search()` tool.
   b. For each relevant search result obtained from that query, extract its 'title', 'url', and 'snippet'.
   c. Construct a ResearchDoc JSON object for each relevant source. For each ResearchDoc:
      - 'url' should be the URL from the search result.
      - 'title' should be the title from the search result.
      - 'content' should be the snippet from the search result. If the snippet is very short, you can briefly elaborate based on the title and URL if confident, but prioritize accuracy. Do not invent content.
      - 'source_type' (optional): If identifiable from the URL or title (e.g., 'linkedin', 'github', 'news_article', 'blog_post', 'company_website'), please populate this field. Otherwise, leave it as null.
      - 'metadata' can be an empty object or contain the source as 'search_result'.
2. Collect ALL ResearchDoc objects generated from ALL search queries into a single, flat JSON array (list).
3. Your FINAL output MUST BE this JSON array of ResearchDoc objects. Each object in the list must validate against the ResearchDoc schema.
   - Example: `[{"url": "example.com/person1", "title": "Person1 Info", "content": "..."}, {"url": "example.com/person2", "title": "Person2 Profile", "content": "..."}]`
   - If a specific query yields no relevant results, simply move to the next query. If ALL queries yield no relevant results after thorough attempts, output an empty JSON array: `[]`.
   - Do not include any other text, explanations, or conversational filler in your output. Only the JSON array."""
    return Agent(
        name="AutonomousResearcher",
        model="gpt-4o-mini",
        instructions=researcher_instructions,
        tools=[
            agent_web_search, # Assuming agent_web_search does not require the client directly
        ],
        output_type=List[ResearchDoc],
        llm=client # Pass the configured client
    )

# ---------------------------------------------------------------------
# 4.  EXPORTS ---------------------------------------------------------
__all__ = [
    "get_hero_agent",
    "get_about_agent",
    "get_features_agent",
    "translate_text",
    "get_researcher_agent",
    "ResearchDoc",
]
