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
import openai
from openai import RateLimitError, APIStatusError
from site_agents import (
    get_hero_agent, get_about_agent, get_features_agent, 
    translate_text, get_researcher_agent, ResearchDoc
)
from agents import Runner
from schemas import HeroSection, AboutSection, FeaturesList, UserProfileData, ResearchOutput
from search_utils import search_web
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy initialization of Firestore client with retry logic
_db = None

def get_db():
    global _db
    if _db is None:
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
                    time.sleep(1)
        
        if _db is None:
            logger.error(f"Failed to initialize Firestore client after {max_retries} attempts: {last_error}")
            if os.getenv("ENVIRONMENT") == "development" or os.getenv("FLASK_ENV") == "development":
                logger.warning("Using mock Firestore client for development environment")
                from unittest.mock import MagicMock
                _db = MagicMock()
            else:
                raise RuntimeError(f"Failed to initialize Firestore client: {last_error}")
    return _db

def get_site_input(uid: str) -> dict | None:
    """Fetches the siteInputDocument for a given UID from Firestore."""
    db = get_db()
    doc_ref = db.collection("siteInputDocuments").document(uid)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return None

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key, max_retries=0)

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

async def run_agent_safely(agent: Any, prompt: str, parent_logger: Optional[logging.Logger] = None, **kwargs) -> Any:
    """Enhanced agent runner with comprehensive error handling."""
    current_logger = parent_logger or logger
    try:
        current_logger.info(f"Running agent '{agent.name if hasattr(agent, 'name') else 'UnknownAgent'}'...")
        
        runner_instance = Runner()
        result = await runner_instance.run(agent, prompt, **kwargs)
        
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

async def do_comprehensive_research(uid: str, timestamp: int | None = None, parent_logger: Optional[logging.Logger] = None):
    """
    Enhanced research function implementing research recommendations:
    1. Researcher-first approach with comprehensive query generation
    2. Enhanced source tracking and validation
    3. Better error handling and fallback mechanisms
    4. Structured output with manifest creation
    """
    current_logger = parent_logger or logger
    research_start_time = time.time()
    
    try:
        current_logger.info(f"Starting comprehensive research for user {uid}")
        
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

        # Enhanced query generation
        company_details = site_input.get('company', {})
        user_profile_for_query_gen = {
            "name": site_input.get("name", ""),
            "title": site_input.get("title") or site_input.get("job_title", ""),
            "bio": site_input.get("bio", ""),
            "professionalBackground": site_input.get("professionalBackground", ""),
            "socialUrls": site_input.get("socialUrls", {}),
            "company": company_details,
            "templateType": site_input.get("templateType", "resume")
        }

        search_queries, direct_urls = await _generate_comprehensive_search_queries(
            user_profile_for_query_gen, oai_client, max_queries=6, parent_logger=current_logger
        )

        if not search_queries and not direct_urls:
            current_logger.warning(f"No search queries or direct URLs generated for user {uid}. Skipping research.")
            await _create_empty_research_manifest(uid, timestamp, current_logger, "no_queries_generated")
            return

        current_logger.info(f"Generated {len(search_queries)} queries and {len(direct_urls)} direct URLs for user {uid}")

        # Research execution with enhanced error handling
        research_docs: List[ResearchDoc] = []
        failed_queries = []
        successful_queries = []
        
        # Check if we should bypass research due to API limits
        BYPASS_RESEARCH_DUE_TO_SERPAPI_CREDITS = os.getenv("BYPASS_SERPAPI_RESEARCH", "false").lower() == "true"

        if BYPASS_RESEARCH_DUE_TO_SERPAPI_CREDITS:
            current_logger.warning("BYPASS: Skipping research agent execution due to SerpAPI configuration.")
            status = "bypassed_serpapi_limits"
        else:
            # Enhanced researcher prompt with better instructions
            researcher_prompt = (
                f"You are conducting comprehensive research for a {user_profile_for_query_gen.get('templateType', 'personal')} site.\n\n"
                f"Profile Summary:\n"
                f"Name: {site_input.get('name', 'N/A')}\n"
                f"Title: {site_input.get('title', 'N/A')}\n"
                f"Company: {company_details.get('name', 'N/A')}\n"
                f"Template Type: {user_profile_for_query_gen.get('templateType', 'resume')}\n\n"
                f"Search Queries: {json.dumps(search_queries)}\n"
                f"Direct URLs to prioritize: {json.dumps(direct_urls)}\n\n"
                f"Instructions:\n"
                f"1. Execute each search query systematically\n"
                f"2. For direct URLs (social profiles), fetch them with high priority\n"
                f"3. Focus on finding factual, verifiable information\n"
                f"4. Prioritize professional achievements, projects, and public presence\n"
                f"5. Return ALL relevant sources as structured ResearchDoc objects\n"
                f"6. Do not fabricate or infer information beyond what sources contain\n\n"
                f"Return a complete list of ResearchDoc objects with accurate source attribution."
            )
            
            researcher_agent_instance = get_researcher_agent()
            
            current_logger.info(f"Running enhanced researcher agent for user {uid}")
            agent_results = await run_agent_safely(
                researcher_agent_instance, 
                researcher_prompt, 
                parent_logger=current_logger
            )

            if agent_results is None:
                current_logger.error(f"Researcher agent failed for user {uid}")
                status = "agent_execution_failed"
            elif not isinstance(agent_results, list):
                current_logger.error(f"Researcher agent returned unexpected type: {type(agent_results)}")
                status = "agent_output_invalid"
            else:
                try:
                    adapter = TypeAdapter(List[ResearchDoc])
                    research_docs = adapter.validate_python(agent_results)
                    status = "completed_successfully"
                    successful_queries = search_queries  # Assume all were attempted
                    current_logger.info(f"Successfully validated {len(research_docs)} research documents")
                except ValidationError as e:
                    current_logger.error(f"Validation error for research documents: {e}")
                    status = "validation_failed"

        # Enhanced data persistence with comprehensive manifest
        db = get_db()
        source_refs = []
        
        # Save research documents
        if research_docs:
            user_research_col_ref = db.collection("research").document(uid).collection("sources")
            
            for i, research_doc in enumerate(research_docs):
                try:
                    source_doc_ref = user_research_col_ref.document()
                    doc_data = research_doc.model_dump()
                    doc_data['timestamp'] = firestore.SERVER_TIMESTAMP
                    doc_data['research_session_id'] = timestamp or int(time.time())
                    
                    source_doc_ref.set(doc_data)
                    source_refs.append(source_doc_ref)
                    current_logger.info(f"Saved source {i+1}/{len(research_docs)}: '{research_doc.title}'")
                except Exception as e:
                    current_logger.error(f"Error saving ResearchDoc '{research_doc.title}': {e}", exc_info=True)

        # Create comprehensive research manifest
        research_duration = time.time() - research_start_time
        
        manifest_data = {
            "uid": uid,
            "status": status,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "original_user_timestamp": timestamp if timestamp else None,
            "research_duration_seconds": round(research_duration, 2),
            "template_type": user_profile_for_query_gen.get('templateType', 'unknown'),
            
            # Query and URL tracking
            "search_queries_generated": search_queries,
            "direct_urls_identified": direct_urls,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            
            # Results tracking
            "total_sources_found": len(research_docs),
            "sources_saved_successfully": len(source_refs),
            "saved_source_ids": [ref.id for ref in source_refs],
            "saved_source_urls": [doc.url for doc in research_docs],
            "source_types_found": list(set([doc.source_type for doc in research_docs if doc.source_type])),
            
            # Quality metrics
            "average_content_length": sum(len(doc.content) for doc in research_docs) / len(research_docs) if research_docs else 0,
            "unique_domains_found": len(set([urllib.parse.urlparse(doc.url).netloc for doc in research_docs])),
            
            # Configuration info
            "bypass_serpapi_enabled": BYPASS_RESEARCH_DUE_TO_SERPAPI_CREDITS,
            "max_queries_configured": 6
        }

        manifest_ref = db.collection("research").document(uid).collection("summary").document("manifest")
        manifest_ref.set(manifest_data)
        
        current_logger.info(f"Research completed for user {uid}. Status: {status}, Duration: {research_duration:.2f}s, Sources: {len(research_docs)}")

    except Exception as e:
        current_logger.error(f"Critical error in do_comprehensive_research for user {uid}: {e}", exc_info=True)
        await _create_error_research_manifest(uid, timestamp, str(e), current_logger)

async def _create_empty_research_manifest(uid: str, timestamp: int | None, logger: logging.Logger, reason: str):
    """Create a manifest when research cannot proceed."""
    try:
        db = get_db()
        manifest_ref = db.collection("research").document(uid).collection("summary").document("manifest")
        
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

async def _create_error_research_manifest(uid: str, timestamp: int | None, error_msg: str, logger: logging.Logger):
    """Create an error manifest when research fails."""
    try:
        db = get_db()
        manifest_ref = db.collection("research").document(uid).collection("summary").document("manifest")
        
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

async def generate_enhanced_site_content(uid: str, languages: List[str], timestamp: int | None = None, parent_logger: Optional[logging.Logger] = None):
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

        site_doc_ref.set(update_data, merge=True)
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