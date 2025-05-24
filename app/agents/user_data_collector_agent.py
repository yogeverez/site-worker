# app/agents/user_data_collector_agent.py

import logging
from agents import Agent, ModelSettings, function_tool, AgentOutputSchema
from app.schemas import UserProfileData
from app.user_data_collector import fetch_and_extract_social_data

logger = logging.getLogger(__name__)

def user_data_collector_agent() -> Agent:
    """
    Agent that collects and enriches user data (e.g., fetching social profiles) and outputs a structured UserProfileData.
    """
    instructions = """You are a User Data Collection Specialist with a focus on EFFICIENCY. 
Complete these steps in a SINGLE TURN if possible:

1. QUICKLY scan the provided user input dictionary for any profile URLs (LinkedIn, GitHub, etc.)
2. For EACH URL found, use fetch_and_extract_social_data ONCE to retrieve profile data
3. IMMEDIATELY combine all fetched data with the original input
4. DIRECTLY output a complete UserProfileData object without intermediate steps

KEY FIELDS to prioritize (only include if data is available):
- name: Full name of the person
- current_title: Current job position
- current_company: Current employer
- skills: List of professional skills (limit to top 5-7)
- education: Brief education history
- bio: 1-2 sentence professional summary

AVOID unnecessary tool calls and reasoning steps. Output the final JSON directly.
"""
    # Wrap the data fetching and summarization functions as tools
    fetch_tool = function_tool(fetch_and_extract_social_data)
    return Agent(
        name="UserDataCollector",
        instructions=instructions + "\nUse the fetched profile data and internal knowledge to produce the final JSON profile.",
        model="gpt-4o-mini",
        tools=[fetch_tool],
        output_type=AgentOutputSchema(UserProfileData, strict_json_schema=False),
        model_settings=ModelSettings(temperature=0.3)
    )
