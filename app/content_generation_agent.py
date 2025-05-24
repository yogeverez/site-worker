"""
Content Generation Agent - Generates website content based on research findings
"""
import logging
import time
import json
from typing import Dict, List, Any, Optional
# Updated imports to use our new structure
from app.agent_types import Agent, ModelSettings, Runner, AgentOutputSchema, function_tool
from app.schemas import HeroSection, AboutSection, FeaturesList, ResearchDoc
# Use our refactored agents
from app.agents.hero_agent import hero_agent
from app.agents.about_agent import about_agent
from app.agents.features_agent import features_agent
from app.database import get_db  # Fixed import path
from google.cloud import firestore
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ContentSection(BaseModel):
    """Structured content section data."""
    section_data: dict = Field(description="The content data for the section")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata")

# Create the implementation function without the decorator
def _get_research_facts_impl(uid: str, session_id: int) -> str:
    """
    Implementation of get_research_facts that can be called directly.
    """
    try:
        db = get_db()
        research_ref = db.collection("research").document(uid).collection("sources")
        
        # Get sources from current session
        query = research_ref.where("research_session_id", "==", session_id)
        sources = query.stream()
        
        facts = []
        for source in sources:
            data = source.to_dict()
            
            # Extract key facts from each source
            fact_entry = {
                "source_url": data.get("url", ""),
                "source_title": data.get("title", ""),
                "key_content": data.get("snippet", ""),
                "source_type": data.get("source_type", "unknown")
            }
            
            # Add specific facts based on content
            content = data.get("content", "")
            if content:
                # Extract notable information
                # This is simplified - in production, use NLP to extract key facts
                fact_entry["extracted_facts"] = content[:500]  # First 500 chars as sample
            
            facts.append(fact_entry)
        
        logger.info(f"Retrieved {len(facts)} research facts for content generation")
        return json.dumps(facts)
        
    except Exception as e:
        logger.error(f"Error retrieving research facts: {str(e)}")
        return json.dumps([])

# Create the decorated version for use as a tool
@function_tool
def get_research_facts(uid: str, session_id: int) -> str:
    """
    Retrieve key facts from research documents for content generation.
    
    Args:
        uid: User ID
        session_id: Session ID
        
    Returns:
        JSON string of key facts extracted from research
    """
    return _get_research_facts_impl(uid, session_id)

def save_content_section(uid: str, section_type: str, content: str, language: str = "en") -> bool:
    """
    Save a generated content section to Firestore.
    
    Args:
        uid: User ID
        section_type: Type of section (hero, about, features)
        content: JSON string of the content data
        language: Language code
        
    Returns:
        True if saved successfully
    """
    try:
        content_data = json.loads(content) if isinstance(content, str) else content
        
        db = get_db()
        doc_ref = db.collection("content").document(uid).collection(language).document(section_type)
        
        doc_ref.set({
            "section_type": section_type,
            "content": content_data,
            "language": language,
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        
        logger.info(f"✅ Saved {section_type} content for {uid} in {language}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving content section: {e}")
        return False

def content_generation_orchestrator() -> Agent:
    """
    Creates an orchestrator agent for content generation that coordinates multiple specialized agents.
    """
    instructions = """
    You are the Content Generation Orchestrator, responsible for creating high-quality website content based on research findings.
    
    Your workflow:
    1. Retrieve research facts using get_research_facts
    2. Coordinate with specialized agents to generate each content section:
       - Hero section: Bold, impactful introduction
       - About section: Professional narrative
       - Features section: Key achievements and skills
    3. Ensure all content is grounded in research findings
    4. Save each section using save_content_section
    
    Guidelines:
    - Never invent information not found in research
    - Maintain professional tone throughout
    - Ensure consistency across all sections
    - Highlight unique achievements and expertise
    - Keep content concise but impactful
    
    Return a summary of what content was generated and saved.
    """
    
    return Agent(
        name="ContentGenerationOrchestrator",
        instructions=instructions,
        model="gpt-4o",
        tools=[get_research_facts, save_content_section],
        model_settings=ModelSettings(temperature=0.6)
    )

# Keep the old function name for backward compatibility
def get_content_generation_orchestrator() -> Agent:
    """
    Backward compatibility wrapper for content_generation_orchestrator()
    """
    return content_generation_orchestrator()

async def generate_content_with_research(
    uid: str, 
    user_input: Dict[str, Any], 
    session_id: int, 
    language: str = "en",
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Generate content for all sections based on research findings.
    
    Args:
        uid: User ID
        user_input: Original user input data
        session_id: Session ID
        language: Target language
        logger: Logger instance
        
    Returns:
        Dictionary with generation results
    """
    current_logger = logger or logging.getLogger(__name__)
    
    try:
        current_logger.info(f"Starting content generation for {uid} in {language}")
        
        # Get research facts
        facts_json = _get_research_facts_impl(uid, session_id)
        facts = json.loads(facts_json) if facts_json else []
        
        if not facts:
            current_logger.warning("No research facts found, generating basic content")
        
        # Prepare context for content generation
        context = {
            "user_input": user_input,
            "research_facts": facts,
            "facts_summary": _summarize_facts(facts)
        }
        
        sections_generated = []
        
        # Generate Hero Section
        try:
            # Use our refactored hero_agent function
            agent = hero_agent()
            hero_prompt = f"""
            Create a hero section for {user_input.get('name', 'this person')}.
            
            User Input:
            - Name: {user_input.get('name', '')}
            - Title: {user_input.get('title', '')}
            - Bio: {user_input.get('bio', '')}
            
            Research Findings:
            {context['facts_summary']}
            
            Create an impactful hero section that captures their essence.
            """
            
            runner = Runner()
            # Pass max_turns directly to run() method
            hero_result = await runner.run(agent, hero_prompt, max_turns=5)
            
            if hero_result:
                hero_data = hero_result.final_output_as(HeroSection, raise_if_incorrect_type=True)
                save_content_section(uid, "hero", hero_data.json(), language)
                sections_generated.append("hero")
                current_logger.info("✅ Hero section generated")
                
        except Exception as e:
            current_logger.error(f"Error generating hero section: {str(e)}")
        
        # Generate About Section
        try:
            # Use our refactored about_agent function
            agent = about_agent()
            about_prompt = f"""
            Create an about section for {user_input.get('name', 'this person')}.
            
            User Input:
            - Name: {user_input.get('name', '')}
            - Title: {user_input.get('title', '')}
            - Bio: {user_input.get('bio', '')}
            - Professional Background: {user_input.get('professionalBackground', '')}
            
            Research Findings:
            {context['facts_summary']}
            
            Write a compelling third-person narrative that tells their professional story.
            """
            
            runner = Runner()
            # Pass max_turns directly to run() method
            about_result = await runner.run(agent, about_prompt, max_turns=5)
            
            if about_result:
                about_data = about_result.final_output_as(AboutSection, raise_if_incorrect_type=True)
                save_content_section(uid, "about", about_data.json(), language)
                sections_generated.append("about")
                current_logger.info("✅ About section generated")
                
        except Exception as e:
            current_logger.error(f"Error generating about section: {str(e)}")
        
        # Generate Features Section
        try:
            # Use our refactored features_agent function
            agent = features_agent()
            features_prompt = f"""
            Create a features section highlighting key achievements for {user_input.get('name', 'this person')}.
            
            User Input:
            - Name: {user_input.get('name', '')}
            - Title: {user_input.get('title', '')}
            - Skills/Background: {user_input.get('professionalBackground', '')}
            
            Research Findings:
            {context['facts_summary']}
            
            Create 3-5 features that showcase their most impressive achievements and capabilities.
            """
            
            runner = Runner()
            # Pass max_turns directly to run() method
            features_result = await runner.run(agent, features_prompt, max_turns=5)
            
            if features_result:
                features_data = features_result.final_output_as(FeaturesList, raise_if_incorrect_type=True)
                save_content_section(uid, "features", features_data.json(), language)
                sections_generated.append("features")
                current_logger.info("✅ Features section generated")
                
        except Exception as e:
            current_logger.error(f"Error generating features section: {str(e)}")
        
        # Update site metadata
        db = get_db()
        site_ref = db.collection("sites").document(uid)
        site_ref.update({
            "metadata.content_generation_complete": True,
            "metadata.sections_generated": sections_generated,
            "metadata.generation_timestamp": firestore.SERVER_TIMESTAMP,
            "metadata.research_backed": len(facts) > 0
        })
        
        return {
            "status": "success",
            "sections_generated": sections_generated,
            "sections_count": len(sections_generated),
            "research_facts_used": len(facts),
            "language": language
        }
        
    except Exception as e:
        current_logger.error(f"Error in content generation: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "sections_generated": [],
            "sections_count": 0
        }

def _summarize_facts(facts: List[Dict[str, Any]]) -> str:
    """Summarize research facts into a concise text."""
    if not facts:
        return "No specific research findings available."
    
    summary_parts = []
    
    # Group facts by type (if fact_type exists)
    achievements = [f for f in facts if f.get("fact_type") == "achievement"]
    projects = [f for f in facts if f.get("fact_type") == "project"]
    education = [f for f in facts if f.get("fact_type") == "education"]
    
    if achievements:
        summary_parts.append(f"Achievements: Found {len(achievements)} recognition(s) or award(s)")
    
    if projects:
        summary_parts.append(f"Projects: Identified {len(projects)} notable project(s)")
    
    if education:
        summary_parts.append(f"Education: Found {len(education)} educational credential(s)")
    
    # If no categorized facts, summarize by source
    if not (achievements or projects or education):
        summary_parts.append(f"Found {len(facts)} research sources with relevant information")
    
    # Add some specific content
    for fact in facts[:3]:  # Top 3 facts
        if fact.get("key_content"):
            summary_parts.append(f"- {fact['key_content'][:100]}...")
        elif fact.get("source_title"):
            summary_parts.append(f"- From: {fact['source_title']}")
    
    return "\n".join(summary_parts)
