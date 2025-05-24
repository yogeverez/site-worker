# app/agents/about_agent.py

from agents import Agent, ModelSettings
from app.schemas import AboutSection

def about_agent() -> Agent:
    """
    Agent for generating the About section (a short biographical paragraph).
    """
    instructions = (
        "You are a professional bio writer. Produce a JSON AboutSection with a third-person paragraph describing the user.\n"
        "- Integrate key achievements, experiences, or roles from the profile and research.\n"
        "- Keep it concise (3-5 sentences) and factual.\n"
        "- Maintain a professional tone, suitable for a personal website.\n"
        "- Output only valid JSON for the AboutSection model."
    )
    return Agent(
        name="AboutSectionAgent",
        instructions=instructions,
        model="gpt-4o-mini",
        output_type=AboutSection,
        model_settings=ModelSettings(temperature=0.7, max_tokens=500)
    )
