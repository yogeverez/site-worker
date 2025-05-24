"""
Research Orchestrator - Coordinates the entire research and content generation pipeline.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from app.database import get_db
from app.process_status_tracker import create_status_tracker
from app.research_manager import ResearchManager

# Define get_site_input function to avoid circular imports
async def get_site_input(uid: str) -> dict:
    """
    Get site input data from Firestore.
    
    Args:
        uid: User ID
        
    Returns:
        Site input data
    """
    db = get_db()
    site_input_ref = db.collection("siteInput").document(uid)
    site_input_doc = site_input_ref.get()
    
    if not site_input_doc.exists:
        logger.warning(f"No site input found for UID: {uid}")
        return {}
        
    return site_input_doc.to_dict() or {}

logger = logging.getLogger(__name__)

async def create_orchestration_pipeline(uid: str, mode: str = "full", languages: List[str] = ["en"]) -> Dict[str, Any]:
    """
    Create and run the orchestration pipeline for site generation.
    
    Args:
        uid: User ID
        mode: Processing mode (full, research_only, content_only)
        languages: List of languages to generate content for
        
    Returns:
        Dictionary with orchestration results
    """
    logger.info(f"[UID: {uid}] Creating orchestration pipeline - Mode: {mode}, Languages: {languages}")
    
    # Create a unique session ID based on timestamp
    session_id = int(time.time())
    
    # Create status tracker
    status_tracker = create_status_tracker(uid, session_id)
    
    # Get site input data
    site_input = await get_site_input(uid)
    
    # Create research manager
    manager = ResearchManager(uid, session_id)
    
    # Run the appropriate pipeline based on mode
    if mode == "full":
        return await manager.run(site_input, languages)
    elif mode == "research_only":
        # Only run the user data collection and research phases
        user_profile = await manager._collect_user_data(site_input)
        research_results = await manager._conduct_research(user_profile)
        return {
            "status": "success",
            "user_profile": user_profile.dict() if user_profile else {},
            "research_summary": research_results
        }
    elif mode == "content_only":
        # Skip research and only generate content
        user_profile = await manager._collect_user_data(site_input)
        content_results = await manager._generate_content(user_profile, {}, languages)
        return {
            "status": "success",
            "user_profile": user_profile.dict() if user_profile else {},
            "content_summary": content_results
        }
    else:
        logger.error(f"[UID: {uid}] Invalid mode: {mode}")
        return {
            "status": "error",
            "error": f"Invalid mode: {mode}"
        }
