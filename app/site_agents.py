"""
site_agents.py – Bridge between old and new agent structure
This file provides backward compatibility by importing from the new agent modules
"""
from __future__ import annotations
import logging
from typing import Any, List, Optional

# Import refactored agents
from app.agents.hero_agent import hero_agent
from app.agents.about_agent import about_agent
from app.agents.features_agent import features_agent
from app.agents.researcher_agent import researcher_agent
from app.agents.content_validator_agent import content_validator_agent
from app.agents.orchestrator_agent import orchestrator_agent
from app.agents.site_generator_agent import site_generator_agent
from app.agents.translator_agent import translator_agent, translate_text
from app.agents.user_data_collector_agent import user_data_collector_agent
from app.agents.research_orchestrator_agent import research_orchestrator_agent
from app.agents.content_generation_orchestrator_agent import content_generation_orchestrator_agent
from app.agents.profile_synthesis_agent import profile_synthesis_agent

# Configure logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1. ENHANCED CONTENT AGENTS -----------------------------------------
def get_hero_agent():
    """Enhanced hero agent with research-grounded instructions."""
    return hero_agent()

def get_about_agent():
    """Enhanced about agent with research integration capabilities."""
    return about_agent()

def get_features_agent():
    """Enhanced features agent with research-backed achievements focus."""
    return features_agent()

# ---------------------------------------------------------------------
# 3. ENHANCED AUTONOMOUS RESEARCHER AGENT ---------------------------
def get_researcher_agent():
    """
    Enhanced autonomous researcher agent implementing research recommendations.
    Features comprehensive search strategy, better source validation, and structured output.
    """
    return researcher_agent()

# ---------------------------------------------------------------------
# 4. PROFILE SYNTHESIS AGENT ----------------------------------------
def get_profile_synthesis_agent():
    """
    Agent that synthesizes research findings into a structured user profile.
    This complements the research agent by organizing findings into categories.
    """
    return profile_synthesis_agent()

# ---------------------------------------------------------------------
# 5. VALIDATION AGENTS -----------------------------------------------
def get_content_validator_agent(content_type: str):
    """
    Creates validation agents for different content types.
    These agents ensure generated content meets quality standards.
    """
    return content_validator_agent(content_type)

# ---------------------------------------------------------------------
# 6. ORCHESTRATOR AGENT ---------------------------------------------
def get_orchestrator_agent():
    """
    Master orchestrator agent that coordinates the entire site generation process.
    Implements the researcher-first approach recommended in the research.
    """
    return orchestrator_agent()

# ---------------------------------------------------------------------
# 7. NEW AGENTS ------------------------------------------------------
# These functions were added based on memory 3fb4c468-319c-40d3-a32f-9be0cb651459
# which mentioned missing agent functions in site_agents.py

def get_site_generator_agent():
    """
    Site generator agent for creating website structure and content.
    """
    return site_generator_agent()

def get_translator_agent():
    """
    Translation agent for localizing content to different languages.
    """
    return translator_agent()

# translate_text function is imported directly from translator_agent.py

# ---------------------------------------------------------------------
# 8. RESEARCH ORCHESTRATOR AGENT ------------------------------------
def get_research_orchestrator_agent():
    """
    Research orchestrator agent that coordinates the research process.
    """
    return research_orchestrator_agent()

# ---------------------------------------------------------------------
# 9. USER DATA COLLECTOR AGENT --------------------------------------
def get_user_data_collector_agent():
    """
    User data collector agent that extracts and processes user data.
    """
    return user_data_collector_agent()

# ---------------------------------------------------------------------
# 10. CONTENT GENERATION ORCHESTRATOR AGENT ------------------------
def get_content_generation_orchestrator():
    """
    Content generation orchestrator agent that coordinates content generation.
    """
    return content_generation_orchestrator_agent()

# ---------------------------------------------------------------------
# 7. EXPORTS ---------------------------------------------------------
__all__ = [
    "get_hero_agent",
    "get_about_agent",
    "get_features_agent",
    "get_researcher_agent",
    "get_profile_synthesis_agent",
    "get_content_validator_agent",
    "get_orchestrator_agent",
    "get_site_generator_agent",
    "get_translator_agent",
    "translate_text",
    "get_research_orchestrator_agent",
    "get_user_data_collector_agent",
    "get_content_generation_orchestrator"
]
