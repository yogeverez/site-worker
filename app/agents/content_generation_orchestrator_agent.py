# app/agents/content_generation_orchestrator_agent.py

import logging
from agents import Agent, ModelSettings, function_tool, handoff
from app.content_generation_agent import get_research_facts, save_content_section
from app.agents.hero_agent import hero_agent
from app.agents.about_agent import about_agent
from app.agents.features_agent import features_agent

logger = logging.getLogger(__name__)

def content_generation_orchestrator_agent(user_id: str, session_id: int, language: str = "en") -> Agent:
    """
    Orchestrator for content generation. Coordinates Hero/About/Features agents and ensures content is saved.
    """
    instructions = f"""You are the Content Generation Orchestrator, responsible for creating high-quality website content in language: {language}.
Workflow:
1. Retrieve key research facts using get_research_facts.
2. Hand off to specialized agents to generate each section:
   - Hero section -> use HeroSectionAgent
   - About section -> use AboutSectionAgent
   - Features section -> use FeaturesAgent
3. Use save_content_section to save each completed section (hero, about, features) for user.
Ensure all content is factually based on the research findings and user profile. Maintain a consistent professional tone across sections.
Do NOT invent details not supported by research or user data.
Finally, confirm completion with a brief summary."""
    # Tools for retrieving research facts and saving content, bound to this user and language
    @function_tool
    def get_facts() -> str:
        """Fetch summarized research facts from Firestore."""
        return get_research_facts(user_id, session_id)
    @function_tool
    def save_section(section_type: str, content: str) -> bool:
        """Save the generated section content to Firestore."""
        return save_content_section(user_id, section_type, content, language)
    return Agent(
        name="ContentGenerationOrchestrator",
        instructions=instructions,
        model="gpt-4o",
        tools=[get_facts, save_section],
        handoffs=[
            hero_agent(), 
            about_agent(), 
            features_agent()
        ],
        model_settings=ModelSettings(temperature=0.6, max_tokens=2000)
    )
