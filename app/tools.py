"""
Enhanced tools and utilities for site worker operations.
"""
import os
import time
import logging
import asyncio
from app import agent_research_tools
import urllib.parse
import threading
import functools
import json
import requests
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from google.cloud import firestore
from pydantic import TypeAdapter, ValidationError
from json import JSONDecodeError
import openai
from openai import RateLimitError, APIStatusError
import datetime

from app.database import get_db
from app.schemas import UserProfileData, ResearchDoc, HeroSection, AboutSection, FeaturesList, ResearchOutput
from app.site_agents import (
    get_researcher_agent, 
    get_profile_synthesis_agent,
    get_site_generator_agent,
    get_translator_agent,
    get_hero_agent, 
    get_about_agent, 
    get_features_agent, 
    translate_text
)
from app.agent_tools import (
    agent_fetch_url, agent_strip_html
)
from app.agent_types import Runner
# RunConfig is not used in our implementation, so we'll remove it
from app.agent_tools import web_search as search_web

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key, max_retries=0)

def convert_firestore_timestamps(data: Any) -> Any:
    """Recursively converts Firestore Timestamp objects to ISO 8601 formatted strings."""
    from google.api_core.datetime_helpers import DatetimeWithNanoseconds
    
    if isinstance(data, DatetimeWithNanoseconds) or isinstance(data, datetime.datetime):
        dt_object = data
        if dt_object.tzinfo is None or dt_object.tzinfo.utcoffset(dt_object) is None:
            dt_object = dt_object.replace(tzinfo=datetime.timezone.utc)
        return dt_object.isoformat()
    elif isinstance(data, dict):
        return {k: convert_firestore_timestamps(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_firestore_timestamps(item) for item in data]
    return data

def get_site_input(uid: str) -> Optional[dict]:
    """Fetches the siteInputDocument for a given UID from Firestore."""
    import os
    db = get_db()
    
    # Check if we're using the mock database
    if os.environ.get("USE_MOCK_DB", "false").lower() == "true" or hasattr(db, "data"):
        # Return a mock site input document for testing
        logger.info(f"Using mock site input document for UID: {uid}")
        return {
            "name": "Nadav Avitan",
            "title": "Civil Engineer",
            "bio": "Experienced civil engineer specializing in construction management and structural design.",
            "professionalBackground": "Over 10 years of experience in the construction industry with expertise in project management and structural analysis.",
            "templateType": "professional",
            "socialUrls": {
                "linkedin": "https://linkedin.com/in/nadav-avitan",
                "website": "https://nadav-construction.com"
            },
            "languages": ["en", "he"],
            "primaryLanguage": "en"
        }
    
    # Regular Firestore lookup
    doc_ref = db.collection("siteInputDocuments").document(uid)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return None

async def _generate_comprehensive_search_queries(user_input: dict, client: openai.OpenAI, max_queries: int = 8, parent_logger: Optional[logging.Logger] = None) -> List[str]:
    """Enhanced query generation based on research recommendations."""
    current_logger = parent_logger or logger
    
    name = user_input.get("name", "")
    title = user_input.get("title") or user_input.get("job_title", "")
    bio = user_input.get("bio", "")
    professional_background = user_input.get("professionalBackground", "")
    template_type = user_input.get("templateType", "resume")
    
    # Enhanced social links processing
    socials = user_input.get("socialUrls", {})
    social_info_parts = []
    direct_urls = []
    
    if isinstance(socials, dict):
        for platform, url in socials.items():
            if url and str(url).strip():
                social_info_parts.append(f"{platform.capitalize()}: {url}")
                # Store direct URLs for priority fetching
                if url.startswith(('http://', 'https://')):
                    direct_urls.append(f"{platform}:{url}")
    
    social_info = ", ".join(social_info_parts)
    company_details = user_input.get('company', {})
    company_name = company_details.get("name", "")

    # Template-specific query generation as recommended in research
    template_specific_queries = {
        "resume": [
            f'"{name}" {company_name} profile',
            f'"{name}" professional achievements',
            f'"{name}" career highlights',
            f'"{name}" awards recognition'
        ],
        "portfolio": [
            f'"{name}" projects portfolio',
            f'"{name}" github repositories',
            f'"{name}" creative work',
            f'"{name}" dribbble behance'
        ],
        "personal": [
            f'"{name}" personal website',
            f'"{name}" blog articles',
            f'"{name}" social media presence',
            f'"{name}" interviews mentions'
        ]
    }

    # Base queries that work for all templates
    base_queries = [
        f'"{name}" linkedin profile',
        f'"{name}" {title}' if title else f'"{name}"',
        f'"{name}" {company_name}' if company_name else f'"{name}" work',
    ]

    # Combine base and template-specific queries
    template_queries = template_specific_queries.get(template_type.lower(), template_specific_queries["resume"])
    all_potential_queries = base_queries + template_queries

    prompt = (
        f"Given the following user profile:\n"
        f"Name: {name}\n"
        f"Title: {title}\n"
        f"Bio: {bio}\n"
        f"Professional Background: {professional_background}\n"
        f"Company: {company_name}\n"
        f"Template Type: {template_type}\n"
        f"Social Links: {social_info}\n\n"
        f"Generate up to {max_queries} highly effective search queries to find comprehensive professional information. "
        f"Focus on queries that would uncover achievements, public presence, projects, and professional persona. "
        f"Prioritize variety and specificity. Consider the template type when crafting queries.\n"
        f"Suggested base queries: {json.dumps(all_potential_queries[:5])}\n"
        f"Return as a JSON object with 'queries' key containing a list of strings and 'direct_urls' key for any social profile URLs to fetch directly.\n"
        f"Example: {{\"queries\": [\"query1\", \"query2\"], \"direct_urls\": [\"{direct_urls[0] if direct_urls else 'linkedin:url'}\"]}} "
    )

    try:
        current_logger.info(f"Generating comprehensive search queries for: {name} (template: {template_type})")
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert research strategist who generates targeted search queries."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        
        response_content = completion.choices[0].message.content
        if response_content:
            try:
                data = json.loads(response_content)
                queries = data.get('queries', [])
                direct_urls_from_llm = data.get('direct_urls', [])
                
                # Combine with our detected direct URLs
                all_direct_urls = list(set(direct_urls + direct_urls_from_llm))
                
                # Ensure all items are strings and limit to max_queries
                queries = [str(q) for q in queries if isinstance(q, str)][:max_queries]
                
                current_logger.info(f"Generated {len(queries)} search queries and {len(all_direct_urls)} direct URLs")
                return queries, all_direct_urls
                
            except json.JSONDecodeError as e:
                current_logger.error(f"Failed to parse JSON response for search queries: {e}")
                return all_potential_queries[:max_queries], direct_urls
        else:
            current_logger.error("LLM returned no content for search queries.")
            return all_potential_queries[:max_queries], direct_urls
            
    except Exception as e:
        current_logger.error(f"Error generating search queries with LLM: {e}", exc_info=True)
        return all_potential_queries[:max_queries], direct_urls

async def run_agent_safely(agent: Any, prompt: str, parent_logger: Optional[logging.Logger] = None, max_turns: int = 15, **kwargs) -> Any:
    """Enhanced agent runner with comprehensive error handling and configurable turn limits."""
    current_logger = parent_logger or logger
    try:
        current_logger.info(f"Running agent '{agent.name if hasattr(agent, 'name') else 'UnknownAgent'}'...")
        
        runner_instance = Runner()
        result = await runner_instance.run(agent, prompt, max_turns=max_turns, **kwargs)
        
        current_logger.info(f"Agent '{agent.name if hasattr(agent, 'name') else 'UnknownAgent'}' completed successfully.")
        return result
        
    except openai.RateLimitError as rle:
        agent_name = agent.name if hasattr(agent, 'name') else 'UnknownAgent'
        current_logger.error(f"OpenAI API rate limit EXCEEDED during run of agent '{agent_name}'. Error: {rle}")
        return None
        
    except openai.APIStatusError as apise:
        agent_name = agent.name if hasattr(agent, 'name') else 'UnknownAgent'
        current_logger.error(f"OpenAI API status error during run of agent '{agent_name}'. Error: {apise}")
        return None
        
    except Exception as e_general:
        agent_name = agent.name if hasattr(agent, 'name') else 'UnknownAgent'
        current_logger.error(f"Unexpected error during run of agent '{agent_name}'. Error: {e_general}", exc_info=True)
        return None

async def do_comprehensive_research(uid: str, timestamp: Optional[int] = None, parent_logger: Optional[logging.Logger] = None):
    """
    Enhanced research function that uses the new orchestration pipeline.
    This now includes user data collection as a pre-research phase.
    """
    current_logger = parent_logger or logger
    session_id = timestamp or int(time.time())
    
    try:
        current_logger.info(f"Starting orchestrated research for UID: {uid}")
        
        # Fetch user input data
        site_input = get_site_input(uid)
        if not site_input:
            current_logger.error(f"No siteInputDocument found for UID: {uid}")
            _create_empty_research_manifest(uid, timestamp, "No user input found", current_logger)
            return
        
        # Import orchestration pipeline
        from research_orchestrator import create_orchestration_pipeline
        
        # Create and run the orchestration pipeline
        pipeline = create_orchestration_pipeline(
            uid=uid,
            session_id=session_id,
            languages=["en"],  # For research phase, we only need English
            parent_logger=current_logger
        )
        
        # Run the pipeline asynchronously
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a task
            task = asyncio.create_task(pipeline.run(site_input))
            results = await task
        else:
            # If not in async context, run with asyncio.run
            results = asyncio.run(pipeline.run(site_input))
        
        # Create research manifest based on results
        research_phase = results.get("phases", {}).get("research", {})
        sources_count = research_phase.get("sources_count", 0)
        
        if sources_count > 0:
            current_logger.info(f"✅ Research completed successfully with {sources_count} sources")
            
            # Create success manifest
            manifest_ref = db.collection("research").document(uid).collection("manifests").document(str(session_id))
            manifest_ref.set({
                "uid": uid,
                "status": "completed",
                "timestamp": firestore.SERVER_TIMESTAMP,
                "session_id": session_id,
                "total_sources_found": sources_count,
                "sources_saved_successfully": sources_count,
                "research_duration_seconds": time.time() - session_id,
                "orchestration_results": results,
                "search_backend": "openai-agents",
                "search_operational": True
            })
        else:
            current_logger.warning(f"Research completed but no sources found")
            _create_empty_research_manifest(uid, timestamp, "No relevant sources found", current_logger)
            
    except Exception as e:
        current_logger.error(f"Error in orchestrated research: {str(e)}", exc_info=True)
        _create_error_research_manifest(uid, timestamp, str(e), current_logger)

# Create async wrapper for backward compatibility
def do_comprehensive_research_sync(uid: str, timestamp: Optional[int] = None, parent_logger: Optional[logging.Logger] = None):
    """Synchronous wrapper for the async research function."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(do_comprehensive_research(uid, timestamp, parent_logger))
    finally:
        loop.close()

async def _create_empty_research_manifest(uid: str, timestamp: int, reason: str, logger: logging.Logger):
    """Create a manifest when research cannot proceed."""
    try:
        db = get_db()
        manifest_ref = db.collection("research").document(uid).collection("manifests").document(str(timestamp))
        
        manifest_data = {
            "uid": uid,
            "status": f"skipped_{reason}",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "original_user_timestamp": timestamp,
            "total_sources_found": 0,
            "reason": reason
        }
        
        manifest_ref.set(manifest_data)
        logger.info(f"Empty research manifest created for user {uid}: {reason}")
    except Exception as e:
        logger.error(f"Failed to create empty research manifest for user {uid}: {e}")

async def _create_error_research_manifest(uid: str, timestamp: int, error_msg: str, logger: logging.Logger):
    """Create an error manifest when research fails."""
    try:
        db = get_db()
        manifest_ref = db.collection("research").document(uid).collection("manifests").document(str(timestamp))
        
        manifest_data = {
            "uid": uid,
            "status": "error_in_processing",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "original_user_timestamp": timestamp,
            "error_message": error_msg,
            "total_sources_found": 0
        }
        
        manifest_ref.set(manifest_data, merge=True)
        logger.info(f"Error research manifest created for user {uid}")
    except Exception as e:
        logger.error(f"Failed to create error research manifest for user {uid}: {e}")

async def generate_enhanced_site_content(uid: str, languages: List[str], timestamp: Optional[int] = None, parent_logger: Optional[logging.Logger] = None):
    """
    Enhanced content generation that utilizes research data effectively.
    Implements research recommendations for grounding content in factual data.
    """
    current_logger = parent_logger or logger
    generation_start_time = time.time()
    
    try:
        current_logger.info(f"Starting enhanced site content generation for user {uid}, languages: {languages}")
        
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

        # Fetch research data to ground content generation
        research_data = await _fetch_research_data(uid, current_logger)
        research_summary = _create_research_summary(research_data)
        
        # Convert Firestore timestamps before serializing
        serializable_site_input = convert_firestore_timestamps(site_input)
        
        # Enhanced base prompt that includes research data
        base_prompt = _create_enhanced_base_prompt(serializable_site_input, research_summary)

        # Initialize agents
        hero_agent_instance = get_hero_agent()
        about_agent_instance = get_about_agent()
        features_agent_instance = get_features_agent()

        # Store generated content
        generated_content = {}
        generation_metrics = {
            "sections_attempted": 0,
            "sections_completed": 0,
            "research_facts_used": len(research_data),
            "content_generation_errors": []
        }

        # Hero Section with enhanced context
        current_logger.info(f"Generating Hero section for {uid} with research context...")
        generation_metrics["sections_attempted"] += 1
        
        hero_prompt = f"{base_prompt}\n\nFocus on creating a compelling hero section that highlights the person's current role and key achievements found in research."
        
        hero_run_result = await run_agent_safely(hero_agent_instance, hero_prompt, parent_logger=current_logger)
        if hero_run_result and hero_run_result.final_output and isinstance(hero_run_result.final_output, HeroSection):
            generated_content["hero"] = hero_run_result.final_output.model_dump()
            generation_metrics["sections_completed"] += 1
            current_logger.info(f"Hero section generated successfully for {uid}")
        else:
            error_msg = f"Failed to generate Hero section for {uid}"
            generation_metrics["content_generation_errors"].append(error_msg)
            current_logger.error(error_msg)

        # About Section with research integration
        current_logger.info(f"Generating About section for {uid} with research integration...")
        generation_metrics["sections_attempted"] += 1
        
        about_prompt = f"{base_prompt}\n\nCreate a comprehensive about section that weaves in the factual achievements and background information found in research. Focus on third-person narrative."
        
        about_run_result = await run_agent_safely(about_agent_instance, about_prompt, parent_logger=current_logger)
        if about_run_result and about_run_result.final_output and isinstance(about_run_result.final_output, AboutSection):
            generated_content["about"] = about_run_result.final_output.model_dump()
            generation_metrics["sections_completed"] += 1
            current_logger.info(f"About section generated successfully for {uid}")
        else:
            error_msg = f"Failed to generate About section for {uid}"
            generation_metrics["content_generation_errors"].append(error_msg)
            current_logger.error(error_msg)

        # Features Section with research-backed achievements
        current_logger.info(f"Generating Features section for {uid} with research-backed content...")
        generation_metrics["sections_attempted"] += 1
        
        features_prompt = f"{base_prompt}\n\nCreate a features list that highlights specific achievements, skills, and accomplishments backed by the research findings. Each feature should be concrete and factual."
        
        features_run_result = await run_agent_safely(features_agent_instance, features_prompt, parent_logger=current_logger)
        if features_run_result and features_run_result.final_output and isinstance(features_run_result.final_output, FeaturesList):
            generated_content["features"] = features_run_result.final_output.model_dump()
            generation_metrics["sections_completed"] += 1
            current_logger.info(f"Features section generated successfully for {uid}")
        else:
            error_msg = f"Failed to generate Features section for {uid}"
            generation_metrics["content_generation_errors"].append(error_msg)
            current_logger.error(error_msg)

        if not generated_content:
            current_logger.error(f"No content was generated for user {uid}. Skipping save and translation.")
            return

        # Save original content with enhanced metadata
        generation_duration = time.time() - generation_start_time
        
        db = get_db()
        site_doc_ref = db.collection("users").document(uid).collection("sites").document("live")

        update_data = {
            "generatedSiteContent": {
                "original": generated_content,
                "generation_metadata": {
                    **generation_metrics,
                    "generation_duration_seconds": round(generation_duration, 2),
                    "research_data_available": len(research_data) > 0,
                    "template_type": site_input.get("templateType", "unknown"),
                    "generation_timestamp": firestore.SERVER_TIMESTAMP
                }
            },
            "status": "content_generated_original",
            "lastUpdated": firestore.SERVER_TIMESTAMP
        }
        if timestamp:
            update_data["requestTimestamp"] = timestamp

        # Always overwrite existing content with new content
        site_doc_ref.set(update_data, merge=False)
        current_logger.info(f"Saved enhanced generated content for user {uid}")

        # Enhanced translation process
        if languages and generated_content:
            await _process_enhanced_translations(uid, languages, generated_content, site_doc_ref, current_logger)

        total_duration = time.time() - generation_start_time
        current_logger.info(f"Enhanced site content generation completed for user {uid} in {total_duration:.2f}s")

    except Exception as e:
        current_logger.error(f"Error during enhanced site content generation for user {uid}: {e}", exc_info=True)
        await _save_content_generation_error(uid, str(e), current_logger)

async def _fetch_research_data(uid: str, logger: logging.Logger) -> List[ResearchDoc]:
    """Fetch research data from Firestore for content grounding."""
    try:
        db = get_db()
        research_docs = []
        
        # Fetch all research sources for the user
        sources_ref = db.collection("research").document(uid).collection("sources")
        docs = sources_ref.limit(20).stream()  # Limit to prevent overwhelming the prompt
        
        for doc in docs:
            try:
                doc_data = doc.to_dict()
                research_doc = ResearchDoc(**doc_data)
                research_docs.append(research_doc)
            except Exception as e:
                logger.warning(f"Failed to parse research doc {doc.id}: {e}")
        
        logger.info(f"Fetched {len(research_docs)} research documents for user {uid}")
        return research_docs
        
    except Exception as e:
        logger.error(f"Error fetching research data for user {uid}: {e}")
        return []

def _create_research_summary(research_data: List[ResearchDoc]) -> str:
    """Create a concise summary of research findings for prompt injection."""
    if not research_data:
        return "No additional research data available."
    
    # Group by source type for better organization
    by_source_type = {}
    for doc in research_data:
        source_type = doc.source_type or "general"
        if source_type not in by_source_type:
            by_source_type[source_type] = []
        by_source_type[source_type].append(doc)
    
    summary_parts = ["=== RESEARCH FINDINGS ==="]
    
    for source_type, docs in by_source_type.items():
        summary_parts.append(f"\n{source_type.upper()} SOURCES:")
        for doc in docs[:3]:  # Limit to top 3 per source type
            summary_parts.append(f"• {doc.title}: {doc.content[:200]}..." if len(doc.content) > 200 else f"• {doc.title}: {doc.content}")
    
    summary_parts.append("\n=== END RESEARCH FINDINGS ===\n")
    return "\n".join(summary_parts)

def _create_enhanced_base_prompt(site_input: dict, research_summary: str) -> str:
    """Create an enhanced base prompt that includes research context."""
    base_info = json.dumps(site_input, indent=2)
    
    enhanced_prompt = f"""USER PROFILE DATA:
{base_info}

{research_summary}

INSTRUCTIONS:
- Use the research findings to enhance and ground your content generation
- Incorporate specific achievements, facts, and details found in the research
- Maintain accuracy - do not fabricate information not present in user input or research
- Focus on factual, verifiable information when available
- If research data conflicts with user input, prioritize user input but note the discrepancy
"""
    
    return enhanced_prompt

async def _process_enhanced_translations(uid: str, languages: List[str], generated_content: dict, site_doc_ref, logger: logging.Logger):
    """Enhanced translation process with better error handling."""
    logger.info(f"Starting enhanced translations for user {uid} into languages: {languages}")
    translated_site_content = {}
    translation_metrics = {"languages_attempted": 0, "languages_completed": 0, "translation_errors": []}

    for lang in languages:
        if lang.lower() in ["en", "english"]:
            logger.info(f"Skipping translation to English for user {uid}")
            continue

        translation_metrics["languages_attempted"] += 1
        logger.info(f"Translating content to {lang} for user {uid}...")
        
        try:
            current_lang_translations = {}
            
            for section, content_item in generated_content.items():
                if not content_item:
                    continue
                
                translated_section = await _translate_content_item(content_item, lang, logger)
                current_lang_translations[section] = translated_section
            
            if current_lang_translations:
                translated_site_content[lang] = current_lang_translations
                translation_metrics["languages_completed"] += 1
                logger.info(f"Completed translations to {lang} for user {uid}")
                
        except Exception as e:
            error_msg = f"Failed to translate content to {lang}: {str(e)}"
            translation_metrics["translation_errors"].append(error_msg)
            logger.error(error_msg, exc_info=True)

    if translated_site_content:
        translation_update_data = {
            "generatedSiteContent": {
                "translations": translated_site_content,
                "translation_metadata": translation_metrics
            },
            "status": "content_generated_translations",
            "lastUpdated": firestore.SERVER_TIMESTAMP
        }
        site_doc_ref.set(translation_update_data, merge=True)
        logger.info(f"Saved enhanced translated content for user {uid}")

async def _translate_content_item(content_item: dict, target_lang: str, logger: logging.Logger) -> dict:
    """Translate a single content item (section) to target language."""
    translated_section = {}
    
    for key, value in content_item.items():
        if isinstance(value, str) and value.strip():
            translated_value = await translate_text(text=value, target_language=target_lang, parent_logger=logger)
            translated_section[key] = translated_value if translated_value and "Error:" not in translated_value else value
            
        elif isinstance(value, list):
            translated_list_items = []
            for item in value:
                if isinstance(item, dict):
                    translated_item = await _translate_content_item(item, target_lang, logger)
                    translated_list_items.append(translated_item)
                else:
                    translated_list_items.append(item)
            translated_section[key] = translated_list_items
        else:
            translated_section[key] = value
    
    return translated_section

async def _save_content_generation_error(uid: str, error_msg: str, logger: logging.Logger):
    """Save content generation error to Firestore."""
    try:
        db = get_db()
        site_doc_ref = db.collection("users").document(uid).collection("sites").document("live")
        error_update_data = {
            "status": "error_content_generation",
            "lastUpdated": firestore.SERVER_TIMESTAMP,
            "errorMessage": error_msg
        }
        site_doc_ref.set(error_update_data, merge=True)
        logger.info(f"Saved content generation error for user {uid}")
    except Exception as db_error:
        logger.error(f"Failed to save content generation error for user {uid}: {db_error}")

# Legacy function aliases for backward compatibility
do_research = do_comprehensive_research
generate_site_content = generate_enhanced_site_content