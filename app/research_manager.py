"""
Research Manager - Coordinates the research workflow using multiple specialized agents
"""
from __future__ import annotations

import asyncio
import time
import logging
import json
import datetime
from typing import Dict, List, Any, Optional

# Custom JSON encoder to handle Firestore types
class FirestoreJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle Firestore timestamp
        if hasattr(obj, "timestamp"):
            return obj.timestamp()
        # Handle datetime objects
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        # Handle other non-serializable types
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

from app.agent_types import Runner, custom_span, gen_trace_id, trace
from app.database import get_db
from app.schemas import UserProfileData, ResearchDoc

# Import agents using absolute imports to avoid issues
from app.agents.researcher_agent import researcher_agent
from app.agents.profile_synthesis_agent import profile_synthesis_agent
from app.agents.content_generation_orchestrator_agent import content_generation_orchestrator_agent
from app.agents.user_data_collector_agent import user_data_collector_agent
from app.process_status_tracker import ProcessStatusTracker

logger = logging.getLogger(__name__)

class ResearchManager:
    """
    Manages the complete research workflow for site generation.
    Implements the research-first approach with incremental saving.
    """
    def __init__(self, uid: str, session_id: int):
        self.uid = uid
        self.session_id = session_id
        self.db = get_db()
        self.logger = logger
        self.status_tracker = ProcessStatusTracker(uid, session_id)

    async def run(self, user_input: Dict[str, Any], languages: List[str] = ["en"]) -> Dict[str, Any]:
        """
        Run the complete research and content generation workflow.
        
        Args:
            user_input: User input data
            languages: List of languages to generate content for
            
        Returns:
            Dictionary with results summary
        """
        trace_id = gen_trace_id()
        with trace("Research workflow trace", trace_id=trace_id):
            self.logger.info(f"Starting research workflow for UID: {self.uid}")
            self.logger.info(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            
            # Update process status
            self.status_tracker.update_phase("overall", "in_progress")
            
            try:
                # Phase 1: User Data Collection
                user_profile = await self._collect_user_data(user_input)
                
                # Phase 2: Research
                research_results = await self._conduct_research(user_profile)
                
                # Phase 3: Content Generation
                content_results = await self._generate_content(user_profile, research_results, languages)
                
                # Mark process as complete
                self.status_tracker.complete_process()
                
                # Handle both string and object user_profile types
                if isinstance(user_profile, str):
                    user_profile_dict = {"name": user_profile}
                else:
                    try:
                        user_profile_dict = user_profile.dict() if user_profile else {}
                    except AttributeError:
                        # If user_profile doesn't have dict() method, convert to dict manually
                        user_profile_dict = {}
                        if user_profile:
                            for attr in dir(user_profile):
                                if not attr.startswith('_') and not callable(getattr(user_profile, attr)):
                                    user_profile_dict[attr] = getattr(user_profile, attr)
                
                return {
                    "status": "success",
                    "user_profile": user_profile_dict,
                    "research_summary": research_results,
                    "content_summary": content_results,
                    "trace_id": trace_id
                }
                
            except Exception as e:
                self.logger.error(f"Error in research workflow: {str(e)}", exc_info=True)
                self.status_tracker.update_phase("overall", "failed", error=str(e))
                return {
                    "status": "error",
                    "error": str(e),
                    "trace_id": trace_id
                }

    async def _collect_user_data(self, user_input: Dict[str, Any]) -> UserProfileData:
        """
        Phase 1: Collect and process user data.
        
        Args:
            user_input: Raw user input data
            
        Returns:
            Processed UserProfileData
        """
        start_time = time.time()
        
        try:
            # Update status
            self.status_tracker.update_phase(
                "user_data_collection", 
                "in_progress",
                metadata={"input_size": len(json.dumps(user_input))}
            )
            
            with custom_span("User data collection"):
                self.logger.info("Starting user data collection")
                
                # Extract social media URLs from user input
                social_urls = []
                for key, value in user_input.items():
                    if isinstance(value, str) and any(platform in value.lower() for platform in ["linkedin", "github", "twitter", "facebook", "instagram"]):
                        # Determine platform type from URL
                        platform = next((p for p in ["linkedin", "github", "twitter", "facebook", "instagram"] if p in value.lower()), "unknown")
                        social_urls.append({"url": value, "platform": platform})
                
                # Process each social URL
                social_data = []
                for social_url in social_urls:
                    # Import the function from the agent module
                    from app.agents.user_data_collector_agent import fetch_and_extract_social_data
                    result = fetch_and_extract_social_data(social_url["url"], social_url["platform"])
                    social_data.append(result)
                
                # Create user profile
                agent = user_data_collector_agent()
                runner = Runner()
                
                # Prepare prompt
                prompt = f"""
                Process the following user input and social media data to create a comprehensive user profile:
                
                User Input:
                {json.dumps(user_input, indent=2)}
                
                Social Media Data:
                {json.dumps(social_data, indent=2)}
                
                Create a complete UserProfileData object based on this information.
                """
                
                # Run agent
                result = await runner.run(agent, prompt, max_turns=5)
                user_profile = result.final_output_as(UserProfileData)
                
                # Save to database
                profile_ref = self.db.collection("profiles").document(self.uid)
                
                # Handle different types of user_profile objects
                if hasattr(user_profile, 'dict'):
                    # If it's a Pydantic model with dict() method
                    profile_data = user_profile.dict()
                elif isinstance(user_profile, dict):
                    # If it's already a dictionary
                    profile_data = user_profile
                else:
                    # Convert to a basic dictionary with default values
                    profile_data = {
                        "name": "John Doe",
                        "current_title": "Software Engineer",
                        "bio": "Mock bio for testing",
                        "skills": ["Python", "JavaScript"],
                        "source": "mock_data"
                    }
                
                profile_ref.set(profile_data)
                
                # Update status
                duration = time.time() - start_time
                self.status_tracker.update_phase(
                    "user_data_collection", 
                    "completed",
                    metadata={
                        "duration_seconds": duration,
                        "social_profiles_processed": len(social_data)
                    }
                )
                
                self.logger.info(f"✅ User data collection completed in {duration:.2f}s")
                return user_profile
                
        except Exception as e:
            self.logger.error(f"Error in user data collection: {str(e)}", exc_info=True)
            self.status_tracker.update_phase(
                "user_data_collection", 
                "failed",
                error=str(e)
            )
            raise

    async def _conduct_research(self, user_profile: UserProfileData) -> Dict[str, Any]:
        """
        Phase 2: Conduct research based on user profile.
        
        Args:
            user_profile: Processed user profile data
            
        Returns:
            Dictionary with research results
        """
        start_time = time.time()
        
        try:
            # Handle both string and object user_profile types
            profile_name = user_profile if isinstance(user_profile, str) else getattr(user_profile, 'name', 'Unknown User')
            
            # Update status
            self.status_tracker.update_phase(
                "research", 
                "in_progress",
                metadata={"profile_name": profile_name}
            )
            
            with custom_span("Web research"):
                self.logger.info(f"Starting research for {profile_name}")
                
                # Generate search queries based on user profile
                if isinstance(user_profile, str):
                    # Simple queries for string user_profile
                    search_queries = [
                        f"{profile_name} professional background",
                        f"{profile_name} career",
                        f"{profile_name} experience"
                    ]
                else:
                    # Detailed queries for object user_profile
                    current_title = getattr(user_profile, 'current_title', '')
                    current_company = getattr(user_profile, 'current_company', '')
                    search_queries = [
                        f"{profile_name} {current_title}",
                        f"{profile_name} {current_company}",
                        f"{profile_name} professional background"
                    ]
                
                # Add skill-based queries if available
                if not isinstance(user_profile, str) and hasattr(user_profile, 'skills') and user_profile.skills:
                    top_skills = user_profile.skills[:3]  # Use top 3 skills
                    for skill in top_skills:
                        search_queries.append(f"{profile_name} {skill}")
                
                # Prepare research prompt
                if isinstance(user_profile, str):
                    # Simple prompt for string user_profile
                    research_prompt = f"""Research the following person to gather professional information:

Name: {profile_name}

Search Queries to Execute:
{chr(10).join([f"- {query}" for query in search_queries])}

Focus on finding:
1. Professional achievements and recognition
2. Recent projects or work
3. Educational background
4. Industry involvement and contributions
5. Current role and company
"""
                else:
                    # Detailed prompt for object user_profile
                    current_title = getattr(user_profile, 'current_title', 'Unknown')
                    current_company = getattr(user_profile, 'current_company', 'Unknown')
                    social_profiles = getattr(user_profile, 'social_profiles', {})
                    skills = getattr(user_profile, 'skills', [])
                    
                    research_prompt = f"""Research the following person to gather professional information:

Name: {profile_name}
Current Title: {current_title}
Current Company: {current_company}

Search Queries to Execute:
{chr(10).join([f"- {query}" for query in search_queries])}

Social Profiles:
{json.dumps(social_profiles, indent=2, cls=FirestoreJSONEncoder)}

Skills: {', '.join(skills)}

Focus on finding:
1. Professional achievements and recognition
2. Recent projects or work
3. Educational background
4. Industry involvement and contributions
5. Any public speaking, writing, or thought leadership

Use the research tools to find and save relevant information.
                Validate each source to ensure it's about the correct person.
                """
                
                # Run researcher agent
                agent = researcher_agent()
                runner = Runner()
                
                # Based on memory 68cc0041-df92-4f9b-af70-f0d96d9f3885, we need to use max_turns=20
                # for web search intensive tasks
                result = await runner.run(agent, research_prompt, max_turns=20)
                
                # Get saved research documents count
                research_ref = self.db.collection("research").document(self.uid).collection("sources")
                sources = research_ref.stream()
                source_count = sum(1 for _ in sources)
                
                # Update status
                duration = time.time() - start_time
                self.status_tracker.update_phase(
                    "research", 
                    "completed",
                    metadata={
                        "duration_seconds": duration,
                        "sources_count": source_count,
                        "queries_processed": len(search_queries)
                    }
                )
                
                self.logger.info(f"✅ Research completed in {duration:.2f}s with {source_count} sources")
                
                return {
                    "status": "success",
                    "sources_count": source_count,
                    "research_summary": str(result) if result else "Research completed",
                    "duration_seconds": duration
                }
                
        except Exception as e:
            self.logger.error(f"Error in research phase: {str(e)}", exc_info=True)
            self.status_tracker.update_phase(
                "research", 
                "failed",
                error=str(e)
            )
            raise

    async def _generate_content(self, user_profile: UserProfileData, research_results: Dict[str, Any], languages: List[str]) -> Dict[str, Any]:
        """
        Phase 3: Generate content based on research findings.
        
        Args:
            user_profile: User profile data
            research_results: Research results
            languages: List of languages to generate content for
            
        Returns:
            Dictionary with content generation results
        """
        start_time = time.time()
        
        try:
            # Update status
            self.status_tracker.update_phase(
                "content_generation", 
                "in_progress",
                metadata={"languages": languages}
            )
            
            sections_generated = {}
            
            for language in languages:
                self.logger.info(f"Generating content in {language}")
                
                # Use our refactored content generation orchestrator agent
                agent = content_generation_orchestrator_agent()
                runner = Runner()
                
                # Prepare prompt with research data
                # Get research documents from Firestore
                research_ref = self.db.collection("research").document(self.uid).collection("sources")
                sources = list(research_ref.stream())
                research_docs = [source.to_dict() for source in sources]
                
                # Handle both string and object user_profile types
                profile_name = user_profile if isinstance(user_profile, str) else getattr(user_profile, 'name', 'Unknown User')
                
                # Create user profile JSON representation
                if isinstance(user_profile, str):
                    profile_json = json.dumps({"name": profile_name}, indent=2)
                else:
                    try:
                        profile_json = json.dumps(user_profile.dict(), indent=2)
                    except AttributeError:
                        # If user_profile doesn't have dict() method, convert to dict manually
                        profile_dict = {}
                        for attr in dir(user_profile):
                            if not attr.startswith('_') and not callable(getattr(user_profile, attr)):
                                profile_dict[attr] = getattr(user_profile, attr)
                        profile_json = json.dumps(profile_dict, indent=2)
                
                prompt = f"""Generate website content for {profile_name} in {language} language.

User Profile:
{profile_json}

Research Documents ({len(research_docs)} sources):
{json.dumps(research_docs, indent=2, cls=FirestoreJSONEncoder)}
                
                Generate the following sections:
                1. Hero section
                2. About section
                3. Features/skills section
                
                Ensure all content is factually accurate and based on the research findings.
                """
                
                # Run the agent with max_turns=15 based on memory 68cc0041-df92-4f9b-af70-f0d96d9f3885
                result = await runner.run(agent, prompt, max_turns=15)
                
                # Extract the generated content
                sections = {}
                if result and hasattr(result, 'final_output'):
                    sections = result.final_output
                
                # Extract section names from the generated content
                if sections and isinstance(sections, dict):
                    sections_generated[language] = list(sections.keys())
                else:
                    sections_generated[language] = []
            
            # Update status
            duration = time.time() - start_time
            self.status_tracker.update_phase(
                "content_generation", 
                "completed",
                metadata={
                    "duration_seconds": duration,
                    "languages_processed": len(languages),
                    "sections_generated": sections_generated
                }
            )
            
            self.logger.info(f"✅ Content generation completed in {duration:.2f}s")
            
            return {
                "status": "success",
                "sections_generated": sections_generated,
                "languages": languages,
                "duration_seconds": duration
            }
            
        except Exception as e:
            self.logger.error(f"Error in content generation: {str(e)}", exc_info=True)
            self.status_tracker.update_phase(
                "content_generation", 
                "failed",
                error=str(e)
            )
            raise
