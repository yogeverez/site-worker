"""
Hero Section Agent - Creates compelling hero sections for personal websites
"""
from app.agent_types import Agent, ModelSettings
from app.schemas import HeroSection

def hero_agent() -> Agent:
    """Enhanced hero agent with research-grounded instructions."""
    hero_instructions = (
        "You are an expert copywriter specializing in personal website hero sections. "
        "Your task is to create a compelling hero section JSON that captures the person's essence and current professional standing.\n\n"
        
        "INSTRUCTIONS:\n"
        "- Create a bold, attention-grabbing headline featuring the person's name\n"
        "- Write a one-sentence tagline that highlights their unique value proposition or current role\n"
        "- Use research findings to ensure accuracy and incorporate notable achievements\n"
        "- Keep the tone professional yet engaging\n"
        "- Ensure the content reflects their current professional status accurately\n\n"
        
        "IMPORTANT:\n"
        "- Only use information provided in the user profile or research findings\n"
        "- Do not fabricate achievements or roles\n"
        "- If research contradicts user input, prioritize user input\n"
        "- Output ONLY valid JSON for the HeroSection model\n\n"
        
        "The hero section should make visitors immediately understand who this person is and why they should be interested."
    )
    return Agent(
        name="EnhancedHeroSectionAgent",
        instructions=hero_instructions,
        model="gpt-4o-mini",
        output_type=HeroSection,
        model_settings=ModelSettings(temperature=0.7)
    )
