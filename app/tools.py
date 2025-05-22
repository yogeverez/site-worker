import os, requests, re
import time
import logging
from bs4 import BeautifulSoup
from google.cloud import firestore
# Import local agents module components
from site_agents import hero_agent, about_agent, features_agent, translate_text
from agents import Runner
from schemas import HeroSection, AboutSection, FeaturesList

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
        doc_ref = db.collection("users").document(uid).collection("siteInput").document("siteInputDocument")
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else {}
    except Exception as e:
        logger.error(f"Error getting site input for user {uid}: {e}")
        return {}

def search_web(query: str, num_results: int = 5) -> list[dict]:
    """Use SerpAPI to search the web and return a list of result dicts (title, link, snippet)."""
    if not SERPAPI_API_KEY:
        logger.warning("SerpAPI key not configured, returning empty results")
        return []
    
    params = {
        "engine": "google", 
        "q": query, 
        "api_key": SERPAPI_API_KEY,
        "num": num_results
    }
    
    # Add retry logic for SerpAPI calls
    max_retries = 3
    retry_count = 0
    results = []
    
    while retry_count < max_retries:
        try:
            logger.info(f"Searching web for '{query}' (attempt {retry_count + 1}/{max_retries})")
            resp = requests.get(
                "https://serpapi.com/search", 
                params=params, 
                timeout=15  # 15 second timeout
            )
            
            if resp.status_code == 200:
                data = resp.json()
                for res in data.get("organic_results", []):
                    link = res.get("link")
                    title = res.get("title")
                    snippet = res.get("snippet")
                    if link and title:
                        results.append({"title": title, "url": link, "snippet": snippet})
                
                logger.info(f"Found {len(results)} search results for '{query}'")
                break  # Success, exit retry loop
            else:
                logger.warning(f"SerpAPI returned status code {resp.status_code}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)  # Wait before retrying
        
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1)  # Wait before retrying
            else:
                logger.error(f"Failed to search web after {max_retries} attempts")
    
    return results

def fetch_page_content(url: str) -> tuple[str, str]:
    """Fetch the page at url and return (title, plaintext_content)."""
    # Add retry logic for fetching page content
    max_retries = 2  # Fewer retries for page content to avoid long delays
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Fetching content from {url} (attempt {retry_count + 1}/{max_retries})")
            resp = requests.get(
                url, 
                timeout=10,  # Shorter timeout for page fetching
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml",
                    "Accept-Language": "en-US,en;q=0.9"
                }
            )
            
            if resp.status_code == 200 and resp.text:
                # Parse HTML content
                soup = BeautifulSoup(resp.text, "html.parser")
                # Get page title
                title = soup.title.string.strip() if soup.title else url
                logger.info(f"Successfully fetched content from {url}")
                break  # Success, exit retry loop
            else:
                logger.warning(f"Failed to fetch content from {url}: HTTP {resp.status_code}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)  # Wait before retrying
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1)  # Wait before retrying
            else:
                logger.error(f"Failed to fetch content after {max_retries} attempts")
                return ("", "")
    
    # If we got here without returning, we have a valid response
    if not hasattr(resp, 'text') or not resp.text:
        return ("", "")
        
    # Parse HTML content
    soup = BeautifulSoup(resp.text, "html.parser")
    # Get page title
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    # Remove script and style tags
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    content = re.sub(r"\s+\s+", " ", text).strip()
    return (title, content)

def do_research(uid: str, timestamp: int = None):
    """Perform web research based on the user's input document and store results in Firestore."""
    try:
        logger.info(f"Starting research for user {uid}")
        # Get basic info about the user (name, title, social URLs)
        input_doc = get_site_input(uid)
        if not input_doc:
            logger.warning(f"No input document found for user {uid}")
            return

        name = input_doc.get("name")
        title = input_doc.get("title") or input_doc.get("job_title")
        social_urls = input_doc.get("socialUrls", {})  # e.g. {"linkedin": "...", "twitter": "..."} or list

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
            existing = db.collection("users").document(uid).collection("research").document(doc_id).get()
            idx = 1
            while existing.exists:
                idx += 1
                doc_id = f"{source}{idx}"
                existing = db.collection("users").document(uid).collection("research").document(doc_id).get()
            # Prepare the document data
            doc_data = {
                "url": url,
                "title": title[:200],  # limit title length
                "content": content[:10000],  # store up to 10k chars of content
                "source": source,
                "timestamp": timestamp or int(time.time()),
                "meta": {}
            }
            db.collection("users").document(uid).collection("research").document(doc_id).set(doc_data)
            logger.info(f"Stored research document {doc_id} for user {uid}")
    except Exception as e:
        logger.error(f"Error in do_research for user {uid}: {e}")
        # Return gracefully instead of crashing the worker
        return

def generate_site_content(uid: str, languages: list[str], timestamp: int = None):
    """Generate website components from research docs and store them (supports multiple languages)."""
    try:
        logger.info(f"Starting content generation for user {uid} in languages: {languages}")
        # Load research documents for the user
        db = get_db()
        docs = db.collection("users").document(uid).collection("research").stream()
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

        # Run each agent to get structured content
        logger.info(f"Running hero agent for user {uid}")
        hero_result = Runner.run_sync(hero_agent, user_prompt)
        hero_data = hero_result.final_output  # HeroSection model instance
        
        logger.info(f"Running about agent for user {uid}")
        about_result = Runner.run_sync(about_agent, user_prompt)
        about_data = about_result.final_output  # AboutSection model
        
        logger.info(f"Running features agent for user {uid}")
        features_result = Runner.run_sync(features_agent, user_prompt)
        features_data = features_result.final_output  # FeaturesList model

        # Convert Pydantic model instances to dict for storing
        hero_json = hero_data.model_dump()
        about_json = about_data.model_dump()
        features_json = features_data.model_dump()

        # Store the generated content for the base language (first in list)
        base_lang = languages[0]
        comp_coll = db.collection("users").document(uid)\
                    .collection("siteContent").document(base_lang)\
                    .collection("components")
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
            comp_coll = db.collection("users").document(uid)\
                        .collection("siteContent").document(lang)\
                        .collection("components")
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
