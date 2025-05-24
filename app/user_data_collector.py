"""
User Data Collector Agent - Pre-processes and summarizes user input data
This module fetches data from user-provided links and creates a comprehensive user profile
"""
import logging
import time
from typing import Dict, List, Any, Optional
from app.agent_types import Agent, function_tool, ModelSettings
from app.schemas import UserProfileData
from app.agent_tools import agent_fetch_url, agent_strip_html, agent_extract_structured_data
from app.database import get_db
import hashlib
import json

logger = logging.getLogger(__name__)

def fetch_and_extract_social_data(url: str, platform: str = "unknown") -> Dict[str, Any]:
    """
    Fetch and extract data from a social media or professional profile URL.
    
    Args:
        url: The URL to fetch
        platform: The platform type (linkedin, github, twitter, etc.)
    
    Returns:
        Dictionary with extracted profile data
    """
    try:
        logger.info(f"Fetching social data from {platform}: {url}")
        
        # Fetch the page content
        html_content = agent_fetch_url(url, max_content_size=15000)
        
        if not html_content or html_content.startswith("Error:"):
            logger.warning(f"Failed to fetch content from {url}")
            return {
                "url": url,
                "platform": platform,
                "status": "failed",
                "error": html_content
            }
        
        # Extract structured data
        structured_data = agent_extract_structured_data(html_content, url)
        
        # Extract clean text content
        clean_text = agent_strip_html(html_content, max_output_size=5000)
        
        # Platform-specific extraction patterns
        extracted_data = {
            "url": url,
            "platform": platform,
            "status": "success",
            "structured_data": structured_data,
            "clean_text": clean_text,
            "extracted_info": {}
        }
        
        # Platform-specific parsing
        if platform == "linkedin":
            # Extract LinkedIn specific data
            if "og:title" in structured_data:
                extracted_data["extracted_info"]["title"] = structured_data["og:title"]
            if "og:description" in structured_data:
                extracted_data["extracted_info"]["description"] = structured_data["og:description"]
                
        elif platform == "github":
            # Extract GitHub specific data
            if "profile" in structured_data:
                extracted_data["extracted_info"] = structured_data["profile"]
                
        elif platform == "twitter" or platform == "x":
            # Extract Twitter/X specific data
            if "og:description" in structured_data:
                extracted_data["extracted_info"]["bio"] = structured_data["og:description"]
        
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error fetching social data from {url}: {str(e)}")
        return {
            "url": url,
            "platform": platform,
            "status": "error",
            "error": str(e)
        }

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

def get_user_data_collector_agent() -> Agent:
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

async def collect_and_save_user_data(uid: str, user_input: Dict[str, Any], session_id: int, logger: logging.Logger) -> Dict[str, Any]:
    """
    Main function to collect, process, and save user data.
    
    Args:
        uid: User ID
        user_input: Raw user input data
        session_id: Current session ID
        logger: Logger instance
        
    Returns:
        Dictionary with collected user profile and status
    """
    try:
        logger.info(f"Starting user data collection for UID: {uid}")
        
        # Update process status
        db = get_db()
        status_ref = db.collection("siteGenerationStatus").document(uid)
        status_ref.set({
            "uid": uid,
            "status": "collecting_user_data",
            "last_updated": time.time(),
            "current_phase": "user_data_collection",
            "session_id": session_id
        }, merge=True)
        
        # Extract social URLs
        social_urls = user_input.get("socialUrls", {})
        social_data = []
        
        # Fetch data from each social URL
        for platform, url in social_urls.items():
            if url and isinstance(url, str) and url.startswith(("http://", "https://")):
                logger.info(f"Fetching data from {platform}: {url}")
                data = fetch_and_extract_social_data(url, platform)
                social_data.append(data)
                
                # Save individual social data
                social_doc_id = hashlib.md5(f"{uid}_{platform}_{url}".encode()).hexdigest()
                social_ref = db.collection("research").document(uid).collection("social_data").document(social_doc_id)
                social_ref.set({
                    **data,
                    "collected_at": time.time(),
                    "session_id": session_id
                })
        
        # Summarize all data
        user_profile = summarize_user_data(user_input, social_data)
        
        # Save summarized profile
        profile_ref = db.collection("research").document(uid).collection("user_profile").document("summary")
        profile_data = user_profile.model_dump()
        profile_data["created_at"] = time.time()
        profile_data["session_id"] = session_id
        profile_data["original_input"] = user_input
        profile_ref.set(profile_data)
        
        # Update status
        status_ref.update({
            "user_data_collected": True,
            "social_profiles_fetched": len(social_data),
            "user_profile_created": True,
            "last_updated": time.time()
        })
        
        logger.info(f"Successfully collected and saved user data for UID: {uid}")
        
        return {
            "status": "success",
            "profile": profile_data,
            "social_data_collected": len(social_data)
        }
        
    except Exception as e:
        logger.error(f"Error collecting user data: {str(e)}", exc_info=True)
        
        # Update error status
        status_ref.update({
            "status": "error",
            "error_phase": "user_data_collection",
            "error_message": str(e),
            "last_updated": time.time()
        })
        
        return {
            "status": "error",
            "error": str(e)
        }

async def collect_user_data(
    uid: str, 
    user_input: Dict[str, Any], 
    session_id: int,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Collect and analyze user data from provided links and input.
    
    Args:
        uid: User ID
        user_input: Raw user input data
        session_id: Session ID for tracking
        logger: Logger instance
        
    Returns:
        Dictionary with collection results and user profile
    """
    current_logger = logger or logging.getLogger(__name__)
    
    try:
        current_logger.info(f"Starting user data collection for {uid}")
        
        # Extract data from social links
        social_urls = user_input.get("socialUrls", {})
        collected_data = []
        
        for platform, url in social_urls.items():
            if url:
                current_logger.info(f"Fetching data from {platform}: {url}")
                data = fetch_and_extract_social_data(url, platform)
                if data:
                    collected_data.append({
                        "platform": platform,
                        "url": url,
                        "data": data
                    })
        
        # Create user profile
        profile_data = {
            "name": user_input.get("name", ""),
            "title": user_input.get("title", ""),
            "bio": user_input.get("bio", ""),
            "professionalBackground": user_input.get("professionalBackground", ""),
            "company": user_input.get("company", {}),
            "socialData": collected_data,
            "templateType": user_input.get("templateType", "resume")
        }
        
        # Save user profile
        saved = True  # Replace with actual save logic
        if saved:
            current_logger.info(f"✅ User data collection completed for {uid}")
            return {
                "status": "success",
                "profile": profile_data,
                "sources_collected": len(collected_data),
                "user_profile": UserProfileData(**profile_data)  # Return as UserProfileData object
            }
        else:
            return {
                "status": "failed",
                "error": "Failed to save user profile",
                "sources_collected": 0
            }
            
    except Exception as e:
        current_logger.error(f"Error in user data collection: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "sources_collected": 0
        }
