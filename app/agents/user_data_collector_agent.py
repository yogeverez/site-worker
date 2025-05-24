"""
User Data Collector Agent - Pre-processes and summarizes user input data
"""
import logging
from typing import Dict, List, Any
from app.agent_types import Agent, ModelSettings
from app.schemas import UserProfileData
from ..user_data_collector import fetch_and_extract_social_data

logger = logging.getLogger(__name__)

def summarize_user_data(user_input: Dict[str, Any], social_data: List[Dict[str, Any]]) -> UserProfileData:
    """
    Summarize all collected user data into a structured profile.
    
    Args:
        user_input: Original user input data
        social_data: List of extracted social media data
    
    Returns:
        Summarized UserProfileData
    """
    profile_data = {
        "name": user_input.get("name", ""),
        "current_title": user_input.get("title") or user_input.get("job_title", ""),
        "current_company": user_input.get("company", ""),
        "bio": user_input.get("bio", ""),
        "skills": [],
        "education": [],
        "achievements": [],
        "projects": [],
        "social_profiles": {},
        "languages_spoken": user_input.get("languages", ["English"]),
        "location": user_input.get("location", "")
    }
    
    # Extract skills from bio and professional background
    professional_background = user_input.get("professionalBackground", "")
    if professional_background:
        # Simple keyword extraction for skills (can be enhanced with NLP)
        skill_keywords = ["python", "javascript", "react", "node", "aws", "docker", "kubernetes", 
                         "machine learning", "data science", "leadership", "management", "design",
                         "marketing", "sales", "finance", "consulting", "engineering"]
        
        combined_text = (professional_background + " " + profile_data["bio"]).lower()
        for keyword in skill_keywords:
            if keyword in combined_text:
                profile_data["skills"].append(keyword.title())
    
    # Process social data
    for social_item in social_data:
        if social_item.get("status") == "success":
            platform = social_item["platform"]
            url = social_item["url"]
            profile_data["social_profiles"][platform] = url
            
            # Extract additional info from social profiles
            extracted_info = social_item.get("extracted_info", {})
            if extracted_info:
                # Update bio if more comprehensive
                if "description" in extracted_info and len(extracted_info["description"]) > len(profile_data["bio"]):
                    profile_data["bio"] = extracted_info["description"]
                
                # Update title if available
                if "title" in extracted_info and not profile_data["current_title"]:
                    profile_data["current_title"] = extracted_info["title"]
    
    return UserProfileData(**profile_data)

def user_data_collector_agent() -> Agent:
    """
    Creates an agent that collects and processes user data before research begins.
    """
    instructions = """
    You are a User Data Collection Specialist. Your task is to:
    
    1. Process the provided user input data
    2. Fetch content from any social media or professional profile URLs provided
    3. Extract relevant information from these profiles
    4. Summarize all collected data into a comprehensive user profile
    
    Focus on extracting:
    - Professional information (title, company, skills)
    - Educational background
    - Notable achievements or projects
    - Social media presence
    - Any other relevant professional details
    
    Use the provided tools to fetch and process URLs, then create a comprehensive UserProfileData summary.
    
    IMPORTANT:
    - Always validate URLs before fetching
    - Handle errors gracefully
    - Don't make assumptions about missing data
    - Preserve the original user input while enriching it with fetched data
    """
    
    return Agent(
        name="UserDataCollector",
        instructions=instructions,
        model="gpt-4o-mini",
        tools=[fetch_and_extract_social_data, summarize_user_data],
        model_settings=ModelSettings(temperature=0.3)
    )
