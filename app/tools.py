import os, requests, re
import time
import logging
import asyncio
import urllib.parse
import threading
import functools
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from google.cloud import firestore
from pydantic import TypeAdapter, ValidationError
from json import JSONDecodeError
import openai # Ensure openai is imported for error types
from openai import RateLimitError, APIStatusError # Import specific error types
# Import local agents module components
from site_agents import (
    get_hero_agent, get_about_agent, get_features_agent, 
    translate_text, get_researcher_agent, ResearchDoc
)
from agents import Runner
from schemas import HeroSection, AboutSection, FeaturesList
from search_utils import search_web
import datetime # Moved here
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy initialization of Firestore client with retry logic
_db = None

def get_db():
    global _db
    if _db is None:
        # Add retry logic for Firestore initialization
        retry_count = 0
        max_retries = 3
        last_error = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Initializing Firestore client (attempt {retry_count + 1}/{max_retries})")
                _db = firestore.Client()
                logger.info("Firestore client initialized successfully")
                break
            except Exception as e:
                retry_count += 1
                last_error = e
                logger.warning(f"Failed to initialize Firestore client (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(1)  # Wait before retrying
        
        if _db is None:
            logger.error(f"Failed to initialize Firestore client after {max_retries} attempts: {last_error}")
            
            # Check if we're in a development environment
            if os.getenv("ENVIRONMENT") == "development" or os.getenv("FLASK_ENV") == "development":
                logger.warning("Using mock Firestore client for development environment")
                from unittest.mock import MagicMock
                _db = MagicMock()
            else:
                # In production, we should fail fast
                raise RuntimeError(f"Failed to initialize Firestore client: {last_error}")
    return _db

# SerpAPI configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_KEY", "")

def get_site_input(uid: str) -> dict | None:
    """Fetches the siteInputDocument for a given UID from Firestore."""
    db = get_db()
    doc_ref = db.collection("siteInputDocuments").document(uid)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return None

# Keep the original fetch_page_content function for backward compatibility
def fetch_page_content(url: str) -> tuple[str, str]:
    """Fetch the page at url and return (title, plaintext_content)."""
    html = requests.get(url, timeout=10).text
    if not html:
        return ("", "")
    
    # Parse HTML content
    soup = BeautifulSoup(html, "html.parser")
    # Get page title
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    # Get content using strip_html
    content = ""
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    content = re.sub(r"\s+\s+", " ", text).strip()
    return (title, content)

import openai # Added for OpenAI client
from firebase_admin import credentials, initialize_app, firestore
# Corrected import path for Timestamp
from google.api_core.datetime_helpers import DatetimeWithNanoseconds # For explicit type checking
# Corrected import based on openai-agents PyPI page
from agents import Runner 
from google.cloud.firestore_v1.base_query import FieldFilter

from site_agents import (
    get_hero_agent, get_about_agent, get_features_agent, 
    translate_text, get_researcher_agent, ResearchDoc
)

# Helper function to get OpenAI client
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key, max_retries=0) # Set max_retries=0

async def _generate_search_queries(user_input: dict, client: openai.OpenAI, max_queries: int = 5, parent_logger: Optional[logging.Logger] = None) -> List[str]:
    """Generates a list of search queries based on user input using an LLM call."""
    current_logger = parent_logger or logger
    name = user_input.get("name", "")
    title = user_input.get("title") or user_input.get("job_title", "")
    bio = user_input.get("bio", "")
    professional_background = user_input.get("professionalBackground", "")
    socials = user_input.get("socialUrls", {})
    social_info_parts = []
    if isinstance(socials, dict):
        for platform, url in socials.items():
            if url and str(url).strip():
                social_info_parts.append(f"{platform.capitalize()}: {url}")
    social_info = ", ".join(social_info_parts)

    prompt = (
        f"Given the following user profile:\n"
        f"Name: {name}\n"
        f"Title: {title}\n"
        f"Bio: {bio}\n"
        f"Professional Background: {professional_background}\n"
        f"Social Links: {social_info}\n\n"
        f"Please generate up to {max_queries} distinct and effective search engine queries to find comprehensive professional information about this person. "
        f"Focus on queries that would uncover their achievements, public presence, projects, and overall professional persona. "
        f"Prioritize variety in the queries. For example, search for their name with company, name with projects, name with specific skills if mentioned, etc.\n"
        f"Return the queries as a JSON list of strings. For example: [\"query1\", \"query2\"]"
    )

    try:
        current_logger.info(f"Generating search queries for: {name}")
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates search queries."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # Request JSON output
            temperature=0.5,
        )
        
        response_content = completion.choices[0].message.content
        if response_content:
            # Assuming the response is a JSON string like '{"queries": ["q1", "q2"]}' or directly a list '["q1", "q2"]'
            # We need to robustly parse this.
            try:
                data = json.loads(response_content)
                if isinstance(data, list):
                    queries = data
                elif isinstance(data, dict) and 'queries' in data and isinstance(data['queries'], list):
                    queries = data['queries']
                else:
                    current_logger.error(f"LLM returned unexpected JSON structure for queries: {response_content}")
                    queries = [] # Fallback to empty list

                # Ensure all items are strings
                queries = [str(q) for q in queries if isinstance(q, str)]
                current_logger.info(f"Generated {len(queries)} search queries: {queries}")
                return queries[:max_queries] # Ensure we don't exceed max_queries
            except json.JSONDecodeError as e:
                current_logger.error(f"Failed to parse JSON response for search queries: {e}. Response: {response_content}")
                return [] # Fallback to empty list if JSON parsing fails
        else:
            current_logger.error("LLM returned no content for search queries.")
            return [] # Fallback to empty list
    except openai.RateLimitError as e:
        current_logger.error(f"OpenAI API rate limit exceeded while generating search queries: {e}")
        return []
    except openai.APIStatusError as e:
        current_logger.error(f"OpenAI API status error while generating search queries: {e}")
        return []
    except Exception as e:
        current_logger.error(f"Error generating search queries with LLM: {e}", exc_info=True)
        return [] # Fallback to empty list


async def run_agent_safely(agent: Any, prompt: str, parent_logger: Optional[logging.Logger] = None, **kwargs) -> Any:
    """Runs an agent with comprehensive error handling for OpenAI API errors."""
    current_logger = parent_logger or logger
    try:
        current_logger.info(f"Running agent '{agent.name if hasattr(agent, 'name') else 'UnknownAgent'}'...")
        
        # Instantiate Runner without arguments
        runner_instance = Runner()
        
        # Call run method on the instance, passing agent and prompt positionally
        result = await runner_instance.run(agent, prompt, **kwargs)
        
        current_logger.info(f"Agent '{agent.name if hasattr(agent, 'name') else 'UnknownAgent'}' completed successfully.")
        return result
    except openai.RateLimitError as rle: # Explicitly qualify openai.RateLimitError
        agent_name = agent.name if hasattr(agent, 'name') else 'UnknownAgent'
        current_logger.error(f"OpenAI API rate limit EXCEEDED during run of agent '{agent_name}'. Error: {rle}")
        return None # Indicate failure due to rate limit
    except openai.APIStatusError as apise: # Explicitly qualify openai.APIStatusError
        agent_name = agent.name if hasattr(agent, 'name') else 'UnknownAgent'
        current_logger.error(f"OpenAI API status error during run of agent '{agent_name}'. Error: {apise}")
        return None # Indicate failure due to API status error
    except Exception as e_general:
        agent_name = agent.name if hasattr(agent, 'name') else 'UnknownAgent'
        error_type_name = type(e_general).__name__
        # Log with exc_info=True to get the full traceback for this unexpected catch
        current_logger.error(f"Unexpected error ({error_type_name}) during run of agent '{agent_name}'. Error: {e_general}", exc_info=True)
        
        # Detailed debugging for RateLimitError and APIStatusError type matching
        if error_type_name == 'RateLimitError' or isinstance(e_general, openai.RateLimitError):
            current_logger.error(f"DEBUG: General handler in run_agent_safely caught something that might be RateLimitError for agent '{agent_name}'.")
            current_logger.error(f"DEBUG: id(openai.RateLimitError) in this scope: {id(openai.RateLimitError)}")
            current_logger.error(f"DEBUG: id(type(e_general)) of caught exception: {id(type(e_general))}")
            current_logger.error(f"DEBUG: type(e_general) is openai.RateLimitError: {type(e_general) is openai.RateLimitError}")
            current_logger.error(f"DEBUG: isinstance(e_general, openai.RateLimitError): {isinstance(e_general, openai.RateLimitError)}")
            if type(e_general) is openai.RateLimitError or isinstance(e_general, openai.RateLimitError):
                current_logger.error(f"CRITICAL: openai.RateLimitError was caught by the GENERAL Exception block in run_agent_safely for agent '{agent_name}'. Specific handler failed.")
        elif error_type_name == 'APIStatusError' or isinstance(e_general, openai.APIStatusError):
            current_logger.error(f"DEBUG: General handler in run_agent_safely caught something that might be APIStatusError for agent '{agent_name}'.")
            current_logger.error(f"DEBUG: id(openai.APIStatusError) in this scope: {id(openai.APIStatusError)}")
            current_logger.error(f"DEBUG: id(type(e_general)) of caught exception: {id(type(e_general))}")
            current_logger.error(f"DEBUG: type(e_general) is openai.APIStatusError: {type(e_general) is openai.APIStatusError}")
            current_logger.error(f"DEBUG: isinstance(e_general, openai.APIStatusError): {isinstance(e_general, openai.APIStatusError)}")
            if type(e_general) is openai.APIStatusError or isinstance(e_general, openai.APIStatusError):
                current_logger.error(f"CRITICAL: openai.APIStatusError was caught by the GENERAL Exception block in run_agent_safely for agent '{agent_name}'. Specific handler failed.")
        return None # Indicate general failure


async def do_research(uid: str, timestamp: int | None = None, parent_logger: Optional[logging.Logger] = None):
    """
    Perform research using researcher_agent.
    1. Generates search queries based on user input.
    2. Calls researcher_agent with these queries.
    3. Saves each found ResearchDoc to Firestore.
    4. Creates a manifest file summarizing the research.
    Stores docs at research/{uid}/sources/{docId}
    Stores manifest at research/{uid}/summary/manifest.json
    """
    current_logger = parent_logger or logger
    try:
        current_logger.info(f"Starting research for user {uid}")
        
        # Get OpenAI client
        try:
            oai_client = get_openai_client()
        except ValueError as e:
            current_logger.error(f"Failed to get OpenAI client for research: {e}")
            return

        site_input = get_site_input(uid)
        if not site_input:
            current_logger.warning(f"No site input found for user {uid}. Skipping research.")
            return

        # Generate search queries
        # Use a default empty dict for company_details if not present
        company_details = site_input.get('company', {})
        user_profile_for_query_gen = {
            "name": site_input.get("name", ""),
            "title": site_input.get("title") or site_input.get("job_title", ""),
            "bio": site_input.get("bio", ""),
            "professionalBackground": site_input.get("professionalBackground", ""),
            "socialUrls": site_input.get("socialUrls", {}),
            "company_name": company_details.get("name", ""),
            "company_description": company_details.get("description", "")
        }

        search_queries = await _generate_search_queries(user_profile_for_query_gen, oai_client, max_queries=2, parent_logger=current_logger)

        if not search_queries:
            current_logger.warning(f"No search queries generated for user {uid}. Skipping researcher agent run.")
            # Optionally, still create an empty manifest or a manifest indicating no research was done
            # For now, just returning.
            return

        current_logger.info(f"Generated {len(search_queries)} search queries for user {uid}: {search_queries}")

        research_docs: List[ResearchDoc] = [] # Initialize as empty
        BYPASS_RESEARCH_DUE_TO_SERPAPI_CREDITS = True # Set to False to re-enable research

        if BYPASS_RESEARCH_DUE_TO_SERPAPI_CREDITS:
            current_logger.warning("TEMPORARY BYPASS: Skipping research agent execution (SerpAPI calls) due to exhausted credits. Proceeding with empty research results.")
        else:
            # Original research execution logic
            researcher_prompt = (
                f"Conduct research based on the following profile and queries:\n"
                f"Profile Summary:\nName: {site_input.get('name', 'N/A')}\nTitle: {site_input.get('title', 'N/A')}\n"
                f"Search Queries: {json.dumps(search_queries)}\n"
                f"Please find and return all relevant sources."
            )
            
            researcher_agent_instance = get_researcher_agent() # Removed client argument
            
            current_logger.info(f"Running researcher agent for user {uid} with {len(search_queries)} queries.")
            agent_results = await run_agent_safely(researcher_agent_instance, researcher_prompt, parent_logger=current_logger)

            if agent_results is None:
                current_logger.error(f"Researcher agent failed or returned no results for user {uid}. Research incomplete.")
                # research_docs remains empty, manifest will reflect this
            elif not isinstance(agent_results, list):
                current_logger.error(f"Researcher agent returned an unexpected type: {type(agent_results)}. Expected list. UID: {uid}")
                # research_docs remains empty, manifest will reflect this
            else:
                try:
                    adapter = TypeAdapter(List[ResearchDoc])
                    research_docs = adapter.validate_python(agent_results) # Populate research_docs here
                except ValidationError as e:
                    current_logger.error(f"Validation error for research documents for user {uid}: {e}. Results: {agent_results}")
                    # research_docs remains empty if validation fails

        # Common logic continues here using the 'research_docs' list (which is empty if bypassed or if agent/validation failed in the 'else' block)
        if not research_docs:
            current_logger.info(f"No research documents were found or validated for user {uid}.")
            # Create a manifest indicating no results, then return
        else:
            current_logger.info(f"Successfully retrieved and validated {len(research_docs)} research documents for user {uid}.")

        db = get_db()
        batch = db.batch()
        source_refs = [] # For manifest

        # Save each ResearchDoc to Firestore
        # research/{uid}/sources/{docId}
        # Using a subcollection 'sources' under a 'research' document for the user
        user_research_col_ref = db.collection("research").document(uid).collection("sources")

        for i, research_doc in enumerate(research_docs):
            try:
                # Create a new document with an auto-generated ID
                source_doc_ref = user_research_col_ref.document()
                source_doc_ref.set(research_doc.model_dump())
                source_refs.append(source_doc_ref)
                current_logger.info(f"  Saved source {i+1}/{len(research_docs)}: '{research_doc.title}' to {source_doc_ref.path}")
            except Exception as e:
                current_logger.error(f"Error saving ResearchDoc '{research_doc.title}' to Firestore: {e}", exc_info=True)
        
        # 4. Create a manifest file
        manifest_ref = db.collection("research").document(uid).collection("summary").document("manifest")
        manifest_data = {
            "uid": uid,
            "status": "completed" if research_docs else "completed_no_sources_found",
            "timestamp": firestore.SERVER_TIMESTAMP, # Use server timestamp for the manifest itself
            "original_user_timestamp": timestamp if timestamp else None, # from pubsub message
            "search_queries_generated": search_queries,
            "saved_sources_count": len(research_docs),
            "saved_source_ids": [ref.id for ref in source_refs],
            "saved_source_urls": [doc.url for doc in research_docs] # Adding URLs for easier reference
        }
        if not research_docs and not search_queries: # If skipped due to no queries
             manifest_data["status"] = "skipped_no_queries"
        elif not research_docs and search_queries: # If queries ran but no sources found
             manifest_data["status"] = "completed_no_sources_found"
        elif not source_refs and research_docs: # If agent returned docs but saving failed for all
            manifest_data["status"] = "completed_save_errors"

        manifest_ref.set(manifest_data)
        current_logger.info(f"Research manifest created/updated for user {uid} at {manifest_ref.path} with status: {manifest_data['status']}")

    except Exception as e:
        current_logger.error(f"Overall error in do_research for user {uid}: {e}", exc_info=True)
        # Attempt to save an error manifest if a critical error occurs early
        try:
            db = get_db()
            manifest_ref = db.collection("research").document(uid).collection("summary").document("manifest")
            # Check if manifest exists to avoid overwriting a more specific status if possible
            # This is a simple check; more sophisticated status management might be needed
            existing_manifest = manifest_ref.get()
            if not existing_manifest.exists or not existing_manifest.to_dict().get("status", "").startswith("completed"):
                error_manifest_data = {
                    "uid": uid,
                    "status": "error_in_processing",
                    "error_message": str(e),
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "original_user_timestamp": timestamp if timestamp else None,
                    "search_queries_generated": locals().get('search_queries', 'not_generated_yet'),
                    "saved_sources_count": len(locals().get('research_docs', [])),
                    "saved_source_ids": locals().get('source_refs', [])
                }
                manifest_ref.set(error_manifest_data, merge=True) # Merge to avoid losing partial data if any
                current_logger.info(f"Error manifest created/updated for user {uid} at {manifest_ref.path}")
        except Exception as manifest_e:
            current_logger.error(f"Failed to save error manifest for user {uid}: {manifest_e}", exc_info=True)

def convert_firestore_timestamps(data: Any) -> Any:
    """
    Recursively converts Firestore Timestamp objects (specifically datetime.datetime
    and DatetimeWithNanoseconds) in a data structure to ISO 8601 formatted strings.
    """
    if isinstance(data, DatetimeWithNanoseconds) or isinstance(data, datetime.datetime):
        dt_object = data # data is already a datetime-like object
        if dt_object.tzinfo is None or dt_object.tzinfo.utcoffset(dt_object) is None:
            dt_object = dt_object.replace(tzinfo=datetime.timezone.utc)
        return dt_object.isoformat()
    elif isinstance(data, dict):
        return {k: convert_firestore_timestamps(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_firestore_timestamps(item) for item in data]
    return data

async def generate_site_content(uid: str, languages: List[str], timestamp: int | None = None, parent_logger: Optional[logging.Logger] = None):
    """Generates site content using various agents and stores it in Firestore."""
    current_logger = parent_logger or logger
    try:
        current_logger.info(f"Starting site content generation for user {uid}, languages: {languages}")
        
        # Get OpenAI client
        try:
            oai_client = get_openai_client()
        except ValueError as e:
            current_logger.error(f"Failed to get OpenAI client for content generation: {e}")
            return

        site_input = get_site_input(uid)
        if not site_input:
            current_logger.warning(f"No site input found for user {uid}. Skipping content generation.")
            return

        # Convert Firestore timestamps before serializing to JSON
        serializable_site_input = convert_firestore_timestamps(site_input)
        base_prompt = json.dumps(serializable_site_input)

        # Initialize agents
        hero_agent_instance = get_hero_agent()
        about_agent_instance = get_about_agent()
        features_agent_instance = get_features_agent()

        # Store generated content (original language)
        generated_content = {}

        # Hero Section
        current_logger.info(f"Generating Hero section for {uid}...")
        hero_run_result = await run_agent_safely(hero_agent_instance, base_prompt, parent_logger=current_logger)
        if hero_run_result and hero_run_result.final_output and isinstance(hero_run_result.final_output, HeroSection):
            generated_content["hero"] = hero_run_result.final_output.model_dump()
            current_logger.info(f"Hero section generated for {uid}.")
        else:
            current_logger.error(f"Failed to generate Hero section for {uid}. Agent run result: {hero_run_result}")

        # About Section
        current_logger.info(f"Generating About section for {uid}...")
        about_run_result = await run_agent_safely(about_agent_instance, base_prompt, parent_logger=current_logger)
        if about_run_result and about_run_result.final_output and isinstance(about_run_result.final_output, AboutSection):
            generated_content["about"] = about_run_result.final_output.model_dump()
            current_logger.info(f"About section generated for {uid}.")
        else:
            current_logger.error(f"Failed to generate About section for {uid}. Agent run result: {about_run_result}")

        # Features Section
        current_logger.info(f"Generating Features section for {uid}...")
        features_run_result = await run_agent_safely(features_agent_instance, base_prompt, parent_logger=current_logger)
        if features_run_result and features_run_result.final_output and isinstance(features_run_result.final_output, FeaturesList):
            generated_content["features"] = features_run_result.final_output.model_dump()
            current_logger.info(f"Features section generated for {uid}.")
        else:
            current_logger.error(f"Failed to generate Features section for {uid}. Agent run result: {features_run_result}")

        if not generated_content:
            current_logger.error(f"No content was generated for user {uid}. Skipping save and translation.")
            return

        # Save original content to Firestore
        # Path: users/{uid}/sites/live (update existing doc)
        db = get_db()
        site_doc_ref = db.collection("users").document(uid).collection("sites").document("live")

        # Prepare data for Firestore update, ensuring not to overwrite siteInputDocument
        update_data = {
            "generatedSiteContent": {
                "original": generated_content
            },
            "status": "content_generated_original",
            "lastUpdated": firestore.SERVER_TIMESTAMP
        }
        if timestamp:
            update_data["requestTimestamp"] = timestamp

        # Update the document, merging with existing data
        site_doc_ref.set(update_data, merge=True) # Use set with merge=True to update or create if not exists
        current_logger.info(f"Saved original generated content for user {uid} to Firestore.")

        # --- Translations ---
        if languages and generated_content: # Only translate if languages are specified and original content exists
            current_logger.info(f"Starting translations for user {uid} into languages: {languages}")
            translated_site_content = {}

            for lang in languages:
                if lang.lower() == "en" or lang.lower() == "english": # Skip if target is English (already original)
                    current_logger.info(f"Skipping translation to English for user {uid} as it's the original language.")
                    # Optionally, copy 'original' to 'en' if strict structure is needed
                    # translated_site_content[lang] = generated_content 
                    continue

                current_logger.info(f"Translating content to {lang} for user {uid}...")
                current_lang_translations = {}
                for section, content_item in generated_content.items():
                    if not content_item: # Skip if original content for this section is missing
                        current_logger.warning(f"Skipping translation of section '{section}' to {lang} for {uid} due to missing original content.")
                        continue
                    
                    translated_section = {}
                    for key, value in content_item.items():
                        if isinstance(value, str) and value.strip():
                            # translated_text is now async, ensure oai_client is passed
                            translated_value = await translate_text(text=value, target_language=lang, parent_logger=current_logger)
                            if translated_value and "Error: Translation failed" not in translated_value:
                                translated_section[key] = translated_value
                            else:
                                current_logger.error(f"Translation failed for text in section '{section}', key '{key}' to {lang} for {uid}. Details: {translated_value}")
                                translated_section[key] = value # Fallback to original value
                        elif isinstance(value, list):
                            # Handle lists (e.g., features in FeaturesList)
                            translated_list_items = []
                            for item in value:
                                if isinstance(item, dict): # Assuming list of dicts like in FeaturesList
                                    translated_item_dict = {}
                                    for item_key, item_value in item.items():
                                        if isinstance(item_value, str) and item_value.strip():
                                            # translated_text is now async
                                            translated_list_value = await translate_text(text=item_value, target_language=lang, parent_logger=current_logger)
                                            if translated_list_value and "Error: Translation failed" not in translated_list_value:
                                                translated_item_dict[item_key] = translated_list_value
                                            else:
                                                current_logger.error(f"Translation failed for list item in section '{section}', key '{item_key}' to {lang} for {uid}. Details: {translated_list_value}")
                                                translated_item_dict[item_key] = item_value # Fallback
                                        else:
                                            translated_item_dict[item_key] = item_value # Non-string or empty string
                                    translated_list_items.append(translated_item_dict)
                                else:
                                    translated_list_items.append(item) # Non-dict item in list
                            translated_section[key] = translated_list_items
                        else:
                            translated_section[key] = value # Non-string, non-list value
                    current_lang_translations[section] = translated_section
                
                if current_lang_translations: # Only add if some translations were made for this language
                    translated_site_content[lang] = current_lang_translations
                    current_logger.info(f"Completed translations to {lang} for user {uid}.")
                else:
                    current_logger.warning(f"No content was translated to {lang} for user {uid}, possibly due to errors or empty original sections.")

            if translated_site_content:
                # Save translations to Firestore
                translation_update_data = {
                    "generatedSiteContent": { # This will merge into existing generatedSiteContent
                        "translations": translated_site_content
                    },
                    "status": "content_generated_translations",
                    "lastUpdated": firestore.SERVER_TIMESTAMP
                }
                site_doc_ref.set(translation_update_data, merge=True)
                current_logger.info(f"Saved translated content for user {uid} to Firestore.")
            else:
                current_logger.info(f"No translations were generated or saved for user {uid}.")
        else:
            current_logger.info(f"No languages specified or no original content to translate for user {uid}.")

        current_logger.info(f"Site content generation and translation (if any) completed for user {uid}.")

    except Exception as e:
        current_logger.error(f"Error during site content generation for user {uid}: {e}", exc_info=True)
        # Update status to reflect error if possible, but avoid new exceptions here
        try:
            db = get_db()
            site_doc_ref = db.collection("users").document(uid).collection("sites").document("live")
            error_update_data = {
                "status": "error_content_generation",
                "lastUpdated": firestore.SERVER_TIMESTAMP,
                "errorMessage": str(e) # Store a brief error message
            }
            site_doc_ref.set(error_update_data, merge=True)
        except Exception as db_error:
            current_logger.error(f"Failed to update Firestore with error status for user {uid}: {db_error}")
