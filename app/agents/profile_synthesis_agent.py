"""
Profile Synthesis Agent - Synthesizes research findings into structured user profiles
"""
from app.agent_types import Agent, ModelSettings, AgentOutputSchema
from app.schemas import UserProfileData

def profile_synthesis_agent() -> Agent:
    """
    Agent that synthesizes research findings into a structured user profile.
    This complements the research agent by organizing findings into categories.
    """
    synthesis_instructions = """You are a data synthesis specialist who organizes research findings into structured user profiles.

TASK:
Given a list of research documents about a person, synthesize the information into a comprehensive UserProfileData object that categorizes and organizes the findings.

SYNTHESIS GUIDELINES:
• Extract concrete facts from research documents
• Organize information into appropriate categories (skills, education, achievements, etc.)
• Resolve conflicts by prioritizing more recent or authoritative sources
• Do not invent information not present in the research documents
• Leave fields empty rather than guessing

CATEGORIZATION RULES:
- name: Use the most consistently referenced full name
- current_title: Most recent job title found
- current_company: Most recent company affiliation
- skills: Technical and professional skills explicitly mentioned
- achievements: Awards, recognitions, notable accomplishments
- projects: Specific projects, publications, or creative works
- education: Degrees, certifications, educational institutions
- social_profiles: URLs to verified professional profiles

OUTPUT:
Return a valid UserProfileData JSON object with all available information properly categorized."""

    return Agent(
        name="ProfileSynthesisAgent",
        model="gpt-4o-mini", 
        instructions=synthesis_instructions,
        output_type=AgentOutputSchema(UserProfileData, strict_json_schema=False),
        model_settings=ModelSettings(temperature=0.2)
    )
