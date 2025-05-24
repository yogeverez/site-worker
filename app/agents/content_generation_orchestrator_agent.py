"""
Content Generation Orchestrator Agent - Coordinates content generation process
"""
from app.agent_types import Agent, ModelSettings, function_tool
from app.content_generation_agent import get_research_facts, save_content_section

def content_generation_orchestrator_agent() -> Agent:
    """
    Creates an orchestrator agent for content generation that coordinates multiple specialized agents.
    """
    instructions = """
    You are the Content Generation Orchestrator, responsible for creating high-quality website content based on research findings.
    
    Your workflow:
    1. Retrieve research facts using get_research_facts
    2. Coordinate with specialized agents to generate each content section:
       - Hero section: Bold, impactful introduction
       - About section: Professional narrative
       - Features section: Key achievements and skills
    3. Ensure all content is grounded in research findings
    4. Save each section using save_content_section
    
    Guidelines:
    - Never invent information not found in research
    - Maintain professional tone throughout
    - Ensure consistency across all sections
    - Highlight unique achievements and expertise
    - Keep content concise but impactful
    
    Return a summary of what content was generated and saved.
    """
    
    return Agent(
        name="ContentGenerationOrchestrator",
        instructions=instructions,
        model="gpt-4o",
        tools=[get_research_facts, save_content_section],
        model_settings=ModelSettings(temperature=0.6)
    )
