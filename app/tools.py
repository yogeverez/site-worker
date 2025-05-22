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

# Thread-safe function to run an agent
async def run_agent_safely(agent, prompt):
    """Run an agent asynchronously using Runner.run()."""
    current_thread = threading.current_thread()
    logger.info(f"Calling Runner.run for agent {agent.name} in thread: {current_thread.name} with prompt: {prompt[:100]}...")
    try:
        # Use the asynchronous Runner.run()
        return await Runner.run(agent, prompt)
    except Exception as e:
        logger.error(f"Exception in Runner.run for agent {agent.name} in thread {current_thread.name}: {e}", exc_info=True)
        raise

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

def get_site_input(uid: str) -> dict:
    """Fetch the user's input document (name, job title, social URLs) from Firestore."""
    try:
        db = get_db()
        # Fetch from the root-level siteInputDocuments collection using uid as document ID
        doc_ref = db.collection("siteInputDocuments").document(uid)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else {}
    except Exception as e:
        logger.error(f"Error getting site input for user {uid}: {e}")
        return {}

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

async def do_research(uid: str, timestamp: int | None = None):
    """
    Perform research using either:
      • researcher_agent  (RESEARCH_MODE=agent  – default)
      • legacy pipeline   (RESEARCH_MODE=legacy)
    Stores docs at research/{uid}/sources/{docId}
    """
    try:
        logger.info(f"Starting research for user {uid}")
        # Determine which research mode to use
        mode = os.getenv("RESEARCH_MODE", "agent").lower()
        
        # Get basic info about the user (name, title, social URLs)
        user_input = get_site_input(uid)
        if not user_input:
            logger.warning(f"No siteInputDocument for {uid}")
            return

        if mode == "legacy":
            logger.info("[research] Using legacy research mode")
            _legacy_research(uid, user_input, timestamp)
            return

        # -------- researcher_agent path ----------
        logger.info("[research] Using agent-based research mode")
        
        # Extract more details from user_input for a richer prompt
        name = user_input.get("name", "")
        title = user_input.get("title") or user_input.get("job_title", "") # Existing logic for title
        location = user_input.get("location", "")
        bio = user_input.get("bio", "")
        professional_background = user_input.get("professionalBackground", "")
        website_goal = user_input.get("websiteGoal", "")
        template = user_input.get("template", "") # e.g., "PersonalBranding"

        socials = user_input.get("socialUrls", {}) # Existing key for social links
        social_lines_list = []
        if isinstance(socials, dict):
            for platform, url in socials.items():
                if url and str(url).strip(): # Ensure URL is not empty or just whitespace
                    social_lines_list.append(f"- {platform.capitalize()}: {str(url).strip()}")
        social_info = "\n".join(social_lines_list) if social_lines_list else "N/A"

        prompt_parts = [
            f"Name: {name}",
            f"Title: {title}",
        ]
        if location:
            prompt_parts.append(f"Location: {location}")
        if bio:
            # For potentially long fields, consider sending only a summary or first N chars if it causes issues.
            # For now, sending full content.
            prompt_parts.append(f"Bio: {bio}") 
        if professional_background:
            prompt_parts.append(f"Professional Background: {professional_background}")
        if website_goal:
            prompt_parts.append(f"Stated Website Goal: {website_goal}")
        if template:
            prompt_parts.append(f"Site Template Type: {template}")

        prompt_parts.append(f"Social Links:\n{social_info}")
        
        # Updated instruction for the agent
        prompt_parts.append(
            "\nBased on all the information above, find up to 8 high-quality sources about this person. "
            "Focus on information relevant to their professional persona, achievements, and public presence. "
            "Return the findings as a JSON array of ResearchDoc objects."
        )
        prompt = "\n\n".join(prompt_parts) # Use double newline for better section separation in the prompt

        # Call the agent (blocking)
        try:
            logger.info(f"Running researcher_agent for {uid}")
            
            # Log the current thread for debugging
            current_thread = threading.current_thread()
            logger.info(f"Running researcher_agent in thread: {current_thread.name}")
            
            # Use our thread-safe runner function
            agent_output = await run_agent_safely(researcher_agent, prompt)
            logger.info(f"Successfully received output from agent for user {uid}. Type: {type(agent_output)}")

            if not agent_output or not hasattr(agent_output, 'final_output') or agent_output.final_output is None:
                logger.warning(f"Agent for {uid} did not return a valid final_output. Raw output: {agent_output}")
                logger.info(f"Falling back to legacy research for {uid} due to missing/invalid agent output.")
                _legacy_research(uid, user_input, timestamp)
                return # Exit after fallback
            
            # If we reach here, agent_output and agent_output.final_output are considered valid
            raw_agent_output = agent_output.final_output 

            final_research_output: List[Dict[str, Any]] = [] # This will hold dicts for Firestore, not used by current agent logic
            parsed_docs: List[ResearchDoc] = [] 

            if isinstance(raw_agent_output, ResearchDoc):
                parsed_docs.append(raw_agent_output)
                logger.info(f"Successfully received ResearchDoc object from agent for user {uid}.")
            elif isinstance(raw_agent_output, list) and all(isinstance(item, ResearchDoc) for item in raw_agent_output):
                parsed_docs.extend(raw_agent_output)
                logger.info(f"Successfully received a list of {len(raw_agent_output)} ResearchDoc objects from agent for user {uid}.")
            elif raw_agent_output is None:
                # This case should have been caught by the check above, but as a safeguard:
                logger.info(f"Researcher agent for user {uid} returned None as final_output (safeguard check).")
                logger.info(f"Falling back to legacy research for {uid} due to None agent output (safeguard check).")
                _legacy_research(uid, user_input, timestamp)
                return # Exit after fallback
            else:
                logger.warning(f"Researcher agent for user {uid} returned unexpected type: {type(raw_agent_output)}. Content: {str(raw_agent_output)[:500]}. Expected ResearchDoc or list of ResearchDoc.")
                logger.info(f"Falling back to legacy research for {uid} due to unexpected agent output type.")
                _legacy_research(uid, user_input, timestamp)
                return # Exit after fallback

            if not parsed_docs:
                logger.warning(f"No ResearchDoc was successfully obtained or parsed for user {uid} from agent output.")
                logger.info(f"Falling back to legacy research for {uid} due to no parsable ResearchDocs.")
                _legacy_research(uid, user_input, timestamp)
                return # Exit after fallback

            # Store research results in Firestore
            db = get_db()
            if db:
                user_research_col = db.collection(f"research/{uid}/sources")
                # Delete existing research documents before adding new ones
                for old_doc in user_research_col.stream():
                    old_doc.reference.delete()
                logger.info(f"Cleared existing research documents for {uid} in collection research/{uid}/sources")
                
                # Use the parsed_docs (list of ResearchDoc objects)
                for i, doc_content in enumerate(parsed_docs):
                    # doc_content is already a ResearchDoc instance
                    doc_id = f"doc_{int(time.time())}_{i}" 
                    research_doc_data = doc_content.model_dump() # Convert Pydantic model to dict
                    research_doc_data["timestamp"] = timestamp or int(time.time())
                    
                    output_doc_ref = db.collection("research").document(uid).collection("sources").document(doc_id)
                    output_doc_ref.set(research_doc_data)
                    logger.info(f"Stored research document {doc_id} for user {uid} at {output_doc_ref.path}")
            else:
                logger.error(f"Firestore client not available. Cannot store research for user {uid}.")

        except Exception as e:
            logger.error(f"Error in agent-based research for user {uid}: {e}", exc_info=True)
            # Fall back to legacy research if agent fails
            logger.info(f"Falling back to legacy research for {uid}")
            _legacy_research(uid, user_input, timestamp)
    except Exception as e:
        logger.error(f"Error in do_research for user {uid}: {e}")
        # Return gracefully instead of crashing the worker
        return

def _legacy_research(uid: str, user_input: dict, timestamp: int | None = None):
    """
    Original search_web → fetch_page_content loop
    (code identical to previous implementation)
    """
    try:
        logger.info(f"Running legacy research for user {uid}")
        name = user_input.get("name")
        title = user_input.get("title") or user_input.get("job_title")
        social_urls = user_input.get("socialUrls", {})  # e.g. {"linkedin": "...", "twitter": "..."} or list

        # Prepare a list of targets to research: start with any given social/profile URLs
        targets = []
        if isinstance(social_urls, dict):
            for source, url in social_urls.items():
                targets.append((source.lower(), url))
        elif isinstance(social_urls, list):
            for url in social_urls:
                domain = "social"
                if "linkedin" in url: domain = "linkedin"
                elif "twitter" in url: domain = "twitter"
                targets.append((domain, url))

        # Also, use web search for the person's name and title to find additional info
        if name:
            query = name if not title else f"{name} {title}"
            results = search_web(query, num_results=5)
            for res in results:
                url = res["url"]
                # Skip if same as an already targeted URL
                if any(url == t[1] for t in targets):
                    continue
                # Identify source name from domain
                domain = re.sub(r'^www\.', '', requests.utils.urlparse(url).netloc)
                source = domain.lower() or "web"
                targets.append((source, url))

        # Fetch each target page and store content
        db = get_db()
        for source, url in targets:
            title, content = fetch_page_content(url)
            if not content:
                continue  # skip if fetch failed or content empty
            # Generate a unique document ID for Firestore
            # Use source name; if already used, append a number
            doc_id = source
            # Ensure unique doc_id in case of duplicates
            doc_id = source
            sources_collection = db.collection("research").document(uid).collection("sources")
            existing = sources_collection.document(doc_id).get()
            idx = 1
            while existing.exists:
                idx += 1
                doc_id = f"{source}{idx}"
                existing = sources_collection.document(doc_id).get()
            # Prepare the document data
            doc_data = {
                "url": url,
                "title": title[:200],  # limit title length
                "content": content[:10000],  # store up to 10k chars of content
                "source_type": source,
                "timestamp": timestamp or int(time.time())
            }
            # Store in research/{uid}/sources collection
            sources_collection = db.collection("research").document(uid).collection("sources")
            sources_collection.document(doc_id).set(doc_data)
            logger.info(f"Stored research document {doc_id} for user {uid} at {sources_collection.document(doc_id).path}")
    except Exception as e:
        logger.error(f"Error in legacy research for user {uid}: {e}")
        # Return gracefully instead of crashing the worker
        return

async def generate_site_content(uid: str, languages: list[str], timestamp: int = None):
    """Generate website components from research docs and store them (supports multiple languages)."""
    try:
        logger.info(f"Starting content generation for user {uid} in languages: {languages}")
        # Load research documents for the user
        db = get_db()
        # Get documents from research/{uid}/sources collection
        sources_collection = db.collection("research").document(uid).collection("sources")
        docs = sources_collection.stream()
        research_docs = [doc.to_dict() for doc in docs]
        
        # Also include the original input data for completeness
        input_doc = get_site_input(uid) or {}
        name = input_doc.get("name", "")
        title = input_doc.get("title") or input_doc.get("job_title") or ""
        basic_info = f"Name: {name}\nTitle: {title}\n"
        
        # Compile context for the agents: brief info plus snippets from research
        context_lines = [basic_info]
        for doc in research_docs:
            src = doc.get("source", "")
            title = doc.get("title", "")
            content = doc.get("content", "")
            # Truncate content to a reasonable length for context
            if content and len(content) > 1000:
                content = content[:1000] + "..."
            context_lines.append(f"Source ({src} - {title}): {content}")
        context_text = "\n\n".join(context_lines).strip()

        # Define a common user prompt for all agents, providing the context info
        user_prompt = (
            f"Use the following information about the user to create website content sections:\n{context_text}\n\n"
            "Now produce the requested section in the required JSON format."
        )

        # Log the current thread for debugging
        current_thread = threading.current_thread()
        logger.info(f"Running content generation in thread: {current_thread.name}")
        
        # Run each agent to get structured content using our thread-safe runner function
        logger.info(f"Running hero agent for user {uid}")
        hero_result = await run_agent_safely(hero_agent, user_prompt)
        hero_data = hero_result.final_output  # HeroSection model instance
        
        logger.info(f"Running about agent for user {uid}")
        about_result = await run_agent_safely(about_agent, user_prompt)
        about_data = about_result.final_output  # AboutSection model
        
        logger.info(f"Running features agent for user {uid}")
        features_result = await run_agent_safely(features_agent, user_prompt)
        features_data = features_result.final_output  # FeaturesList model

        # Convert Pydantic model instances to dict for storing
        hero_json = hero_data.model_dump()
        about_json = about_data.model_dump()
        features_json = features_data.model_dump()

        # Store the generated content for the base language (first in list)
        base_lang = languages[0]
        comp_coll = db.collection("siteContent").document(uid)\
                    .collection(base_lang)
        comp_coll.document("hero").set({"timestamp": timestamp or int(time.time()), **hero_json})
        comp_coll.document("about").set({"timestamp": timestamp or int(time.time()), **about_json})
        comp_coll.document("featuresList").set({"timestamp": timestamp or int(time.time()), **features_json})
        logger.info(f"Stored site content for base language '{base_lang}' for user {uid}")

        # Translate to additional languages if any
        for lang in languages[1:]:
            logger.info(f"Translating content to {lang} for user {uid}")
            # Deep copy the original JSONs to avoid in-place modifications
            hero_trans = _translate_component_dict(hero_json, lang)
            about_trans = _translate_component_dict(about_json, lang)
            features_trans = _translate_component_dict(features_json, lang)
            comp_coll = db.collection("siteContent").document(uid)\
                        .collection(lang)
            comp_coll.document("hero").set({"timestamp": timestamp or int(time.time()), **hero_trans})
            comp_coll.document("about").set({"timestamp": timestamp or int(time.time()), **about_trans})
            comp_coll.document("featuresList").set({"timestamp": timestamp or int(time.time()), **features_trans})
            logger.info(f"Stored translated content for language '{lang}' for user {uid}")
    except Exception as e:
        logger.error(f"Error in generate_site_content for user {uid}: {e}")
        # Return gracefully instead of crashing the worker
        return

def _translate_component_dict(comp_dict: dict, target_lang: str) -> dict:
    """Recursively translate all string values in a component JSON dict to the target language."""
    translated = {}
    for key, value in comp_dict.items():
        # Do not translate the key names, only the values
        if isinstance(value, str):
            translated_value = translate_text(value, target_lang)
            translated[key] = translated_value
        elif isinstance(value, list):
            # Translate each element in list (e.g., list of features)
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    new_list.append(_translate_component_dict(item, target_lang))
                elif isinstance(item, str):
                    new_list.append(translate_text(item, target_lang))
                else:
                    new_list.append(item)
            translated[key] = new_list
        elif isinstance(value, dict):
            translated[key] = _translate_component_dict(value, target_lang)
        else:
            translated[key] = value  # keep other types as is
    return translated
