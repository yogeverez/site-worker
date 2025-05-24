"""
Site Generator Agent - Creates website structure and content
"""
from app.agent_types import Agent, ModelSettings

def site_generator_agent() -> Agent:
    """
    Site generator agent for creating website structure and content.
    """
    instructions = """You are a professional website generator that creates modern, engaging personal websites.
    
    Your task is to generate complete website content including:
    - HTML structure with responsive design
    - Professional styling
    - Optimized content layout
    - SEO-friendly elements
    
    Focus on creating clean, modern designs that effectively showcase the user's professional profile and achievements."""

    return Agent(
        name="SiteGeneratorAgent",
        instructions=instructions,
        model="gpt-4o-mini",
        model_settings=ModelSettings(
            temperature=0.3,
            max_tokens=4000
        )
    )
