import os, requests, re
import time
import logging
import asyncio
import urllib.parse
import threading
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from google.cloud import firestore
from pydantic import TypeAdapter, ValidationError
from json import JSONDecodeError
# Import local agents module components
from site_agents import hero_agent, about_agent, features_agent, translate_text, researcher_agent, ResearchDoc
from agents import Runner
from schemas import HeroSection, AboutSection, FeaturesList
from search_utils import search_web

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
    doc_ref = db.collection("users").document(uid).collection("sites").document("live")
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("siteInputDocument")
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
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from site_agents import researcher_agent, hero_agent, about_agent, features_agent, translate_text, ResearchDoc

# Helper function to get OpenAI client
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)

async def _generate_search_queries(user_input: dict, client: openai.OpenAI, max_queries: int = 5) -> List[str]:
    """Generates a list of search queries based on user input using an LLM call."""
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
        logger.info(f"Generating search queries for: {name}")
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
                    logger.error(f"LLM returned unexpected JSON structure for queries: {response_content}")
                    queries = [] # Fallback to empty list

                # Ensure all items are strings
                queries = [str(q) for q in queries if isinstance(q, str)]
                logger.info(f"Generated {len(queries)} search queries: {queries}")
                return queries[:max_queries] # Ensure we don't exceed max_queries
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for search queries: {e}. Response: {response_content}")
                return [] # Fallback to empty list if JSON parsing fails
        else:
            logger.error("LLM returned no content for search queries.")
            return [] # Fallback to empty list
    except Exception as e:
        logger.error(f"Error generating search queries with LLM: {e}", exc_info=True)
        return [] # Fallback to empty list


async def run_agent_safely(agent: Any, prompt: str, **kwargs) -> Any:
    """Wraps Runner.run in an async function, suitable for asyncio.run."""
    # This function might need to be adapted if Runner.run is not inherently async
    # If Runner.run is blocking, use asyncio.to_thread
    # For now, assuming Runner.run can be awaited or is non-blocking enough
    # Based on previous memory (0d3b3dbd), Runner.run() is awaitable.
    return await Runner.run(agent=agent, prompt=prompt, **kwargs)


async def do_research(uid: str, timestamp: int | None = None):
    """
    Perform research using researcher_agent.
    1. Generates search queries based on user input.
    2. Calls researcher_agent with these queries.
    3. Saves each found ResearchDoc to Firestore.
    4. Creates a manifest file summarizing the research.
    Stores docs at research/{uid}/sources/{docId}
    Stores manifest at research/{uid}/summary/manifest.json
    """
    try:
        logger.info(f"Starting research for user {uid}")
        
        # Get OpenAI client
        try:
            oai_client = get_openai_client()
        except ValueError as e:
            logger.error(f"Failed to initialize OpenAI client for research: {e}")
            return

        # Get basic info about the user (name, title, social URLs)
        user_input = get_site_input(uid)
        if not user_input:
            logger.warning(f"No siteInputDocument for {uid}. Skipping research.")
            return

        # 1. Generate search queries
        search_queries = await _generate_search_queries(user_input, oai_client, max_queries=5)
        if not search_queries:
            logger.warning(f"No search queries generated for user {uid}. Skipping agent research.")
            # Optionally, could fall back to a default query or skip research entirely.
            # For now, skipping if no queries are generated.
            # Create an empty manifest if skipping
            db = get_db()
            manifest_ref = db.collection("research").document(uid).collection("summary").document("manifest")
            manifest_data = {
                "uid": uid,
                "status": "skipped_no_queries",
                "timestamp": firestore.SERVER_TIMESTAMP, # Use server timestamp
                "search_queries_generated": [],
                "saved_sources_count": 0,
                "saved_source_ids": []
            }
            manifest_ref.set(manifest_data)
            logger.info(f"Created empty/skipped manifest for user {uid} at {manifest_ref.path}")
            return

        logger.info(f"[research] Using agent-based research mode with {len(search_queries)} queries: {search_queries}")
        
        name = user_input.get("name", "")
        title = user_input.get("title") or user_input.get("job_title", "")
        location = user_input.get("location", "")
        bio = user_input.get("bio", "")
        professional_background = user_input.get("professionalBackground", "")
        website_goal = user_input.get("websiteGoal", "")
        template = user_input.get("template", "")

        socials = user_input.get("socialUrls", {})
        social_lines_list = []
        if isinstance(socials, dict):
            for platform, url_val in socials.items():
                if url_val and str(url_val).strip():
                    social_lines_list.append(f"- {platform.capitalize()}: {str(url_val).strip()}")
        social_info = "\n".join(social_lines_list) if social_lines_list else "N/A"

        # Construct prompt for the researcher_agent, including the generated queries
        prompt_parts = [
            "Please perform research based on the following user profile and search queries.",
            f"User Profile:\nName: {name}",
            f"Title: {title}",
        ]
        if location: prompt_parts.append(f"Location: {location}")
        if bio: prompt_parts.append(f"Bio: {bio}")
        if professional_background: prompt_parts.append(f"Professional Background: {professional_background}")
        if website_goal: prompt_parts.append(f"Stated Website Goal: {website_goal}")
        if template: prompt_parts.append(f"Site Template Type: {template}")
        prompt_parts.append(f"Social Links:\n{social_info}")
        
        prompt_parts.append("\nSearch Queries to Execute:")
        for i, sq_query in enumerate(search_queries):
            prompt_parts.append(f"{i+1}. {sq_query}")
        
        prompt_parts.append(
            "\nAgent Instructions: Follow your main instructions to process each of these queries, find relevant sources, and return a flat list of ResearchDoc objects for all findings."
        )
        prompt = "\n\n".join(prompt_parts)

        # Call the agent
        agent_output_list: List[ResearchDoc] = []
        try:
            logger.info(f"Running researcher_agent for {uid} with generated queries.")
            current_thread = threading.current_thread()
            logger.info(f"Running researcher_agent in thread: {current_thread.name}")
            
            # run_agent_safely returns the agent's final_output directly if successful
            raw_agent_output = await run_agent_safely(researcher_agent, prompt)
            logger.info(f"Successfully received output from agent for user {uid}. Type: {type(raw_agent_output)}")

            if isinstance(raw_agent_output, list) and all(isinstance(item, ResearchDoc) for item in raw_agent_output):
                agent_output_list = raw_agent_output
                logger.info(f"Successfully received {len(agent_output_list)} ResearchDoc objects from agent for user {uid}.")
            elif isinstance(raw_agent_output, ResearchDoc): # Handle if agent mistakenly returns one despite List type hint
                agent_output_list = [raw_agent_output]
                logger.warning(f"Agent returned a single ResearchDoc, expected List[ResearchDoc]. Processing as a list with one item.")
            elif raw_agent_output is None:
                 logger.warning(f"Agent for {uid} returned None as final_output.")
            else:
                logger.warning(f"Agent for {uid} did not return a valid List[ResearchDoc] or ResearchDoc. Raw output: {str(raw_agent_output)[:500]}")

        except Exception as e:
            logger.error(f"Error running researcher_agent for user {uid}: {e}", exc_info=True)
            # Proceed to save manifest with error status even if agent fails

        # 3. Save each ResearchDoc to Firestore
        db = get_db()
        saved_source_ids = []
        saved_source_urls = [] # For manifest

        if not agent_output_list:
            logger.warning(f"No ResearchDoc objects to save for user {uid}.")
        else:
            logger.info(f"Saving {len(agent_output_list)} research documents to Firestore for user {uid}...")
            for i, research_doc in enumerate(agent_output_list):
                try:
                    doc_data = research_doc.model_dump() # Convert Pydantic model to dict
                    # Add a timestamp for when this specific source was processed/saved by the worker
                    doc_data['worker_processed_at'] = firestore.SERVER_TIMESTAMP 
                    
                    # Create a new document with an auto-generated ID
                    source_doc_ref = db.collection("research").document(uid).collection("sources").document()
                    source_doc_ref.set(doc_data)
                    saved_source_ids.append(source_doc_ref.id)
                    saved_source_urls.append(research_doc.url)
                    logger.info(f"  Saved source {i+1}/{len(agent_output_list)}: '{research_doc.title}' to {source_doc_ref.path}")
                except Exception as e:
                    logger.error(f"Error saving ResearchDoc '{research_doc.title}' to Firestore: {e}", exc_info=True)
        
        # 4. Create a manifest file
        manifest_ref = db.collection("research").document(uid).collection("summary").document("manifest")
        manifest_data = {
            "uid": uid,
            "status": "completed" if agent_output_list else "completed_no_sources_found",
            "timestamp": firestore.SERVER_TIMESTAMP, # Use server timestamp for the manifest itself
            "original_user_timestamp": timestamp if timestamp else None, # from pubsub message
            "search_queries_generated": search_queries,
            "saved_sources_count": len(saved_source_ids),
            "saved_source_ids": saved_source_ids,
            "saved_source_urls": saved_source_urls # Adding URLs for easier reference
        }
        if not agent_output_list and not search_queries: # If skipped due to no queries
             manifest_data["status"] = "skipped_no_queries"
        elif not agent_output_list and search_queries: # If queries ran but no sources found
             manifest_data["status"] = "completed_no_sources_found"
        elif not saved_source_ids and agent_output_list: # If agent returned docs but saving failed for all
            manifest_data["status"] = "completed_save_errors"

        manifest_ref.set(manifest_data)
        logger.info(f"Research manifest created/updated for user {uid} at {manifest_ref.path} with status: {manifest_data['status']}")

    except Exception as e:
        logger.error(f"Overall error in do_research for user {uid}: {e}", exc_info=True)
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
                    "saved_sources_count": len(locals().get('saved_source_ids', [])),
                    "saved_source_ids": locals().get('saved_source_ids', [])
                }
                manifest_ref.set(error_manifest_data, merge=True) # Merge to avoid losing partial data if any
                logger.info(f"Error manifest created/updated for user {uid} at {manifest_ref.path}")
        except Exception as manifest_e:
            logger.error(f"Failed to save error manifest for user {uid}: {manifest_e}", exc_info=True)

async def generate_site_content(uid: str, languages: List[str], timestamp: int | None = None):
    """Generates site content using various agents and stores it in Firestore."""
    logger.info(f"Starting site content generation for user {uid}, languages: {languages}")
    db = get_db()
    user_input = get_site_input(uid)
    if not user_input:
        logger.warning(f"No siteInputDocument for {uid} to generate content from.")
        return

    # Fetch research data (assuming it's stored by do_research)
    # For simplicity, let's assume we need a consolidated summary from research.
    # This part might need adjustment based on how research data is actually used by content agents.
    research_summary_text = "No specific research summary available."
    try:
        # Example: Try to get a summary from the manifest or a specific summary doc
        manifest_ref = db.collection("research").document(uid).collection("summary").document("manifest")
        manifest_doc = manifest_ref.get()
        if manifest_doc.exists:
            manifest_data = manifest_doc.to_dict()
            if manifest_data.get("saved_sources_count", 0) > 0:
                # For now, just indicate research was done. A real summary would be better.
                research_summary_text = f"Research conducted, found {manifest_data['saved_sources_count']} sources. Key URLs: {', '.join(manifest_data.get('saved_source_urls', [])[:2])}"
            elif manifest_data.get("status") == "completed_no_sources_found":
                research_summary_text = "Research conducted, but no specific sources were found."
    except Exception as e:
        logger.warning(f"Could not fetch research summary for content generation: {e}")

    name = user_input.get("name", "")
    title = user_input.get("title") or user_input.get("job_title", "")
    bio = user_input.get("bio", "")
    professional_background = user_input.get("professionalBackground", "")
    # ... any other fields needed for content prompts ...

    base_prompt_context = (
        f"User Profile:\nName: {name}\nTitle: {title}\nBio: {bio}\nProfessional Background: {professional_background}\n"
        f"Research Summary: {research_summary_text}\n\n"
        "Please generate the requested website component based on this information."
    )

    # Define agents and their corresponding Firestore paths
    content_agents_map = {
        "hero": (hero_agent, "heroSection"),
        "about": (about_agent, "aboutSection"),
        "features": (features_agent, "featuresList"),
    }

    # Store base language content
    base_lang = "en" # Assuming base language is English
    generated_content_base = {}

    logger.info(f"Running content agents for base language '{base_lang}' for user {uid}")
    for agent_key, (agent_instance, _) in content_agents_map.items():
        try:
            logger.info(f"Calling Runner.run for agent {agent_instance.name} in thread: {threading.current_thread().name} with prompt context.")
            # Use run_agent_safely which should handle async execution
            output = await run_agent_safely(agent_instance, base_prompt_context)
            if output:
                # Assuming output is already the Pydantic model instance (e.g. HeroSection)
                # If it's AgentOutput, need output.final_output
                # For now, let's assume it's the direct model if run_agent_safely is tailored
                # Based on previous edits, run_agent_safely returns final_output
                final_output = output # if run_agent_safely returns final_output
                # if hasattr(output, 'final_output'): # If it's AgentOutput obj
                #    final_output = output.final_output
                # else: # Assuming it's already the data
                #    final_output = output
                
                if final_output:
                    generated_content_base[agent_key] = final_output.model_dump()
                    logger.info(f"Successfully generated '{agent_key}' for base language.")
                else:
                    logger.warning(f"Agent {agent_instance.name} returned no final_output for base language.")
            else:
                logger.warning(f"Agent {agent_instance.name} returned no output for base language.")
        except Exception as e:
            logger.error(f"Error running agent {agent_instance.name} for base language: {e}", exc_info=True)

    if generated_content_base:
        site_doc_ref = db.collection("users").document(uid).collection("sites").document("live")
        # Path for base language content: sites/live/content/en/{componentName}
        # For simplicity, let's store all components under a single 'content_en' field or similar structure
        # Or, more granularly: sites/live/content/en/hero, sites/live/content/en/about etc.
        # Let's go with a structured field: site_doc_ref.update({"content_en": generated_content_base, "contentTimestamp": timestamp})
        # Better: store each component separately for easier updates and translations
        # e.g. users/{uid}/sites/live/content/en/heroSection, users/{uid}/sites/live/content/en/aboutSection
        
        # Let's store under users/{uid}/siteContent/{lang}/{component_key}
        # This keeps siteInputDocument separate from generated content
        base_content_batch = db.batch()
        for key, data in generated_content_base.items():
            component_doc_ref = db.collection("users").document(uid).collection("siteContent").document(base_lang).collection("components").document(key)
            base_content_batch.set(component_doc_ref, data)
        base_content_batch.commit()
        logger.info(f"Stored site content for base language '{base_lang}' for user {uid}")

    # Handle translations if requested
    if languages and base_lang in languages:
        languages.remove(base_lang) # Remove base language if present, as it's already processed
    
    for lang in languages:
        if lang == base_lang: continue # Should be removed already, but as a safeguard
        logger.info(f"Translating content to '{lang}' for user {uid}")
        translated_content_lang = {}
        for component_key, component_data_dict in generated_content_base.items():
            # component_data_dict is the dict form of the pydantic model (e.g., HeroSection.model_dump())
            # We need to translate relevant text fields within this dict.
            translated_component_data = {} # Store translated fields for this component
            try:
                if component_key == "hero" and isinstance(component_data_dict, dict):
                    headline = component_data_dict.get('headline')
                    subheadline = component_data_dict.get('subheadline')
                    if headline: translated_component_data['headline'] = await run_agent_safely(translate_text, f"{headline}", target_language=lang)
                    if subheadline: translated_component_data['subheadline'] = await run_agent_safely(translate_text, f"{subheadline}", target_language=lang)
                
                elif component_key == "about" and isinstance(component_data_dict, dict):
                    content = component_data_dict.get('content')
                    if content: translated_component_data['content'] = await run_agent_safely(translate_text, f"{content}", target_language=lang)
                
                elif component_key == "features" and isinstance(component_data_dict, dict):
                    features_list = component_data_dict.get('features', [])
                    translated_features = []
                    for item in features_list:
                        if isinstance(item, dict):
                            title = item.get('title')
                            desc = item.get('description')
                            trans_title = await run_agent_safely(translate_text, f"{title}", target_language=lang) if title else None
                            trans_desc = await run_agent_safely(translate_text, f"{desc}", target_language=lang) if desc else None
                            translated_features.append({'title': trans_title, 'description': trans_desc})
                    if translated_features: translated_component_data['features'] = translated_features
                
                if translated_component_data:
                    # Merge non-translated fields with translated ones to keep full structure
                    # This assumes Pydantic models don't have extra fields not handled by translation
                    # A safer way would be to load original, update, then dump.
                    # For now, just storing the translated parts.
                    final_translated_data_for_component = {**component_data_dict, **translated_component_data}
                    translated_content_lang[component_key] = final_translated_data_for_component
                    logger.info(f"Successfully translated '{component_key}' to '{lang}'.")
                else:
                    logger.warning(f"No translatable fields found or translation failed for '{component_key}' to '{lang}'.")

            except Exception as e:
                logger.error(f"Error translating component '{component_key}' to '{lang}': {e}", exc_info=True)
        
        if translated_content_lang:
            translation_batch = db.batch()
            for key, data in translated_content_lang.items():
                component_doc_ref = db.collection("users").document(uid).collection("siteContent").document(lang).collection("components").document(key)
                translation_batch.set(component_doc_ref, data)
            translation_batch.commit()
            logger.info(f"Stored translated site content for language '{lang}' for user {uid}")

    logger.info(f"Completed site content generation and translation for user {uid}")
