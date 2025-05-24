# app/agents/user_data_collector_agent.py

import logging
from agents import Agent, ModelSettings, function_tool
from app.schemas import UserProfileData
from app.user_data_collector import fetch_and_extract_social_data, summarize_user_data

logger = logging.getLogger(__name__)

def user_data_collector_agent() -> Agent:
    """
    Agent that collects and enriches user data (e.g., fetching social profiles) and outputs a structured UserProfileData.
    """
    instructions = """You are a User Data Collection Specialist. 
Your task:
1. Process the provided user input dictionary.
2. Fetch content from any social media or professional profile URLs provided.
3. Extract relevant information from these profiles.
4. Summarize all information into a comprehensive user profile.

Focus on professional details: current title, company, skills, education, notable achievements, projects, and social presence. 
Use the tools to fetch and parse profiles, then combine the results with the original input. 
Output a complete user profile in JSON format."""
    # Wrap the data fetching and summarization functions as tools
    fetch_tool = function_tool(fetch_and_extract_social_data)
    summarize_tool = function_tool(summarize_user_data)
    return Agent(
        name="UserDataCollector",
        instructions=instructions,
        model="gpt-4o-mini",
        tools=[fetch_tool, summarize_tool],
        output_type=UserProfileData,
        model_settings=ModelSettings(temperature=0.3)
    )
