# app/agents/hero_agent.py

from agents import Agent, ModelSettings, AgentOutputSchema
from app.schemas import HeroSection

def hero_agent() -> Agent:
    """
    Agent for generating the Hero section of the site (headline and tagline).
    """
    instructions = (
        "You are an expert copywriter for personal website hero sections. "
        "Write a JSON object for HeroSection with a bold headline (often the name) and a one-line subheadline (tagline) capturing the user's essence.\n"
        "- Use information from the provided profile and research facts.\n"
        "- The headline should be attention-grabbing and include the name.\n"
        "- The subheadline should highlight a unique value proposition or role, in one concise sentence.\n"
        "- Maintain a professional and engaging tone.\n"
        "- Output only valid JSON for the HeroSection model."
    )
    return Agent(
        name="HeroSectionAgent",
        instructions=instructions,
        model="gpt-4o-mini",
        output_type=AgentOutputSchema(HeroSection, strict_json_schema=False),
        model_settings=ModelSettings(temperature=0.7, max_tokens=300)
    )
