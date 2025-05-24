"""
Research Orchestrator Agent - Coordinates all phases of the research process
"""
import json
import logging
from app.agent_types import Agent, function_tool, ModelSettings

logger = logging.getLogger(__name__)

@function_tool
def validate_source_with_user_data(source_json: str, user_profile_json: str) -> str:
    """
    Validate if a research source is relevant based on the collected user profile data.
    
    Args:
        source_json: JSON string of research document to validate  
        user_profile_json: JSON string of user profile data to validate against
    
    Returns:
        JSON string with validation result and reasoning
    """
    try:
        # Parse the JSON strings
        source_data = json.loads(source_json)
        user_profile_data = json.loads(user_profile_json)
        
        # Extract key identifiers from user profile
        user_name = user_profile_data.get('name', '').lower()
        user_title = user_profile_data.get('current_title', '').lower()
        user_company = user_profile_data.get('current_company', '').lower()
        user_skills = [skill.lower() for skill in user_profile_data.get('skills', [])]
        social_profiles = user_profile_data.get('social_profiles', {})
        
        # Check content relevance
        content_lower = source_data.get('content', '').lower()
        title_lower = source_data.get('title', '').lower()
        source_url = source_data.get('url', '')
        
        relevance_indicators = {
            "name_match": user_name and user_name in content_lower,
            "title_match": user_title and any(word in content_lower for word in user_title.split()),
            "company_match": user_company and user_company in content_lower,
            "skills_match": any(skill in content_lower for skill in user_skills),
            "social_profile_match": any(
                profile_url in source_url
                for profile_url in social_profiles.values()
                if profile_url
            )
        }
        
        # Calculate relevance score
        relevance_score = sum(1 for match in relevance_indicators.values() if match)
        is_relevant = relevance_score >= 2  # At least 2 indicators must match
        
        # Special case: if it's their social profile, it's always relevant
        if relevance_indicators["social_profile_match"]:
            is_relevant = True
            relevance_score = 5
        
        reasoning = []
        if relevance_indicators["name_match"]:
            reasoning.append(f"Name '{user_name}' found in content")
        if relevance_indicators["title_match"]:
            reasoning.append(f"Job title '{user_title}' matches content")
        if relevance_indicators["company_match"]:
            reasoning.append(f"Company '{user_company}' mentioned")
        if relevance_indicators["skills_match"]:
            matched_skills = [skill for skill in user_skills if skill in content_lower]
            reasoning.append(f"Skills matched: {', '.join(matched_skills[:3])}")
        if relevance_indicators["social_profile_match"]:
            reasoning.append("Source is user's social profile")
        
        result = {
            "is_relevant": is_relevant,
            "relevance_score": relevance_score,
            "reasoning": " | ".join(reasoning) if reasoning else "No specific matches found",
            "indicators": relevance_indicators
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error validating source: {e}")
        return json.dumps({
            "is_relevant": False,
            "relevance_score": 0,
            "reasoning": f"Validation error: {str(e)}",
            "indicators": {}
        })

def research_orchestrator_agent() -> Agent:
    """
    Creates the master orchestrator agent that coordinates all phases of the research process.
    """
    instructions = """
    You are the Research Orchestrator, responsible for coordinating the entire research and content generation process.
    
    Your workflow consists of these phases:
    
    1. **User Data Collection Phase**:
       - Collect and process all user-provided data
       - Fetch content from social media links
       - Create a comprehensive user profile summary
       - Save this data for later validation use
    
    2. **Research Phase**:
       - Generate comprehensive search queries based on user data
       - Execute web searches to find relevant information
       - Validate each source against the user profile
       - Save only relevant sources to the database
    
    3. **Content Generation Phase**:
       - Use the research findings to generate website content
       - Ensure content is grounded in factual research data
       - Create engaging and accurate content sections
    
    4. **Translation Phase** (if needed):
       - Translate generated content to requested languages
       - Maintain quality and context in translations
    
    For each phase:
    - Update the process status in real-time
    - Handle errors gracefully
    - Log important metrics
    - Ensure data is saved incrementally
    
    Use the provided tools to validate sources against the user profile data.
    Focus on quality over quantity - better to have fewer highly relevant sources than many irrelevant ones.
    """
    
    return Agent(
        name="ResearchOrchestrator",
        instructions=instructions,
        model="gpt-4o",
        tools=[validate_source_with_user_data],
        model_settings=ModelSettings(temperature=0.3)
    )
