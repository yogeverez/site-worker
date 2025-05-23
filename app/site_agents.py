"""site_agents.py – all OpenAI agents & function-tools"""
from __future__ import annotations
import os, re, requests, urllib.parse
import openai
from openai import RateLimitError, APIStatusError # Import for error handling
from agents import Agent, function_tool, ModelSettings # Ensure ModelSettings is explicitly imported
from schemas import (
    HeroSection, AboutSection, FeaturesList,  # content schemas
    ResearchDoc                               # NEW – research schema
)
from typing import Any, List
from agent_tool_impl import agent_web_search, agent_fetch_url, agent_strip_html
from schemas import ResearchDoc
from firebase_admin import firestore
import logging

# Configure logging (ensure logger is available)
logger = logging.getLogger(__name__)

# Remove global API key initialization: openai.api_key = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------
# 1.  CONTENT AGENTS --------------------------------------------------
def get_hero_agent() -> Agent:
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
        model_settings=ModelSettings() # Use ModelSettings class
    )

def get_about_agent() -> Agent:
    about_instructions = (
        "You write an 'About Me' section for a personal site in third person. Summarize the user's background, skills, and interests in a single paragraph. "
        "Output only valid JSON for the AboutSection model."
    )
    return Agent(
        name="AboutSectionAgent",
        instructions=about_instructions,
        model="gpt-4o-mini",
        output_type=AboutSection,
        model_settings=ModelSettings() # Use ModelSettings class
    )

def get_features_agent() -> Agent:
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
        model_settings=ModelSettings() # Use ModelSettings class
    )

# ---------------------------------------------------------------------
# 2.  TRANSLATOR function-tool ----------------------------------------
@function_tool
async def translate_text(text: str, target_language: str) -> str:
    if not text or not target_language:
        logger.warning("translate_text called with empty text or target_language.")
        return text if text else "" # Return original text or empty if original is None/empty

    prompt = f"Translate the following text to {target_language}. Output ONLY the translated text, with no additional explanations, commentary, or quotation marks. If the text is a name, a brand, a number, or an email address that should not be translated, return it as is. Text to translate: \"{text}\""
    
    try:
        completion = await openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful translation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Lower temperature for more deterministic translation
        )
        translated_text = completion.choices[0].message.content.strip()
        logger.info(f"Successfully translated text to {target_language}. Original: '{text[:30]}...', Translated: '{translated_text[:30]}...'" )
        return translated_text
    except openai.RateLimitError as rle: # Explicitly qualify openai.RateLimitError
        logger.error(f"OpenAI API rate limit EXCEEDED during translation to {target_language}. Error: {rle}")
        return f"Error: Translation failed (RateLimitError). Original: {text}"
    except openai.APIStatusError as apise: # Explicitly qualify openai.APIStatusError
        logger.error(f"OpenAI API status error during translation to {target_language}. Error: {apise}")
        return f"Error: Translation failed (APIStatusError). Original: {text}"
    except Exception as e_general:
        error_type_name = type(e_general).__name__
        # Log with exc_info=True to get the full traceback for this unexpected catch
        logger.error(f"Unexpected error ({error_type_name}) during translation to {target_language}. Error: {e_general}", exc_info=True)
        
        # Detailed debugging for RateLimitError type matching
        if error_type_name == 'RateLimitError' or isinstance(e_general, openai.RateLimitError):
            logger.error(f"DEBUG: General handler caught something that might be RateLimitError.")
            logger.error(f"DEBUG: id(openai.RateLimitError) in this scope: {id(openai.RateLimitError)}")
            logger.error(f"DEBUG: id(type(e_general)) of caught exception: {id(type(e_general))}")
            logger.error(f"DEBUG: type(e_general) is openai.RateLimitError: {type(e_general) is openai.RateLimitError}")
            logger.error(f"DEBUG: isinstance(e_general, openai.RateLimitError): {isinstance(e_general, openai.RateLimitError)}")
            if type(e_general) is openai.RateLimitError or isinstance(e_general, openai.RateLimitError):
                 logger.error("CRITICAL: openai.RateLimitError was caught by the GENERAL Exception block in translate_text. Specific handler failed.")
                 return f"Error: Translation failed (RateLimitError caught by general handler). Original: {text}"
        return f"Error: Translation failed (Unexpected {error_type_name}). Original: {text}"

# ---------------------------------------------------------------------
# 3.  AUTONOMOUS RESEARCHER AGENT -------------------------------------
def get_researcher_agent() -> Agent:
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
        model_settings=ModelSettings() # Use ModelSettings class
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
