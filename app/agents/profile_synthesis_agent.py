# app/agents/profile_synthesis_agent.py

from agents import Agent, ModelSettings
from app.schemas import UserProfileData

def profile_synthesis_agent() -> Agent:
    """
    Agent that synthesizes all research findings into an updated user profile (UserProfileData).
    """
    instructions = """You are a data synthesis specialist. You will receive research findings about a person and the original profile.
Organize the information into a structured UserProfileData JSON, updating fields like achievements, projects, and skills with any new info from research.
Ensure consistency and factual accuracy.
Output the merged profile as JSON."""
    return Agent(
        name="ProfileSynthesisAgent",
        instructions=instructions,
        model="gpt-4o-mini",
        output_type=UserProfileData,
        model_settings=ModelSettings(temperature=0.3, max_tokens=800)
    )
