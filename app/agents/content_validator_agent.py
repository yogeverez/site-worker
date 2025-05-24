"""
Content Validation Agent - Ensures generated content meets quality standards
"""
from app.agent_types import Agent, ModelSettings
from app.schemas import HeroSection, AboutSection, FeaturesList

def content_validator_agent(content_type: str) -> Agent:
    """
    Creates validation agents for different content types.
    These agents ensure generated content meets quality standards.
    """
    validation_instructions = f"""You are a content validation specialist for {content_type} sections.

VALIDATION CRITERIA:
• Ensure content is factual and grounded in provided information
• Check that tone and style are appropriate for professional websites
• Verify that required fields are present and properly formatted
• Flag any potential inaccuracies or unsupported claims

CORRECTION GUIDELINES:
• Fix formatting issues and ensure JSON validity
• Remove or flag unverifiable statements
• Ensure content length is appropriate for the section type
• Maintain professional tone throughout

If content passes validation, return it unchanged.
If corrections are needed, return the corrected version.
If content is fundamentally flawed, suggest regeneration."""

    output_type_map = {
        "hero": HeroSection,
        "about": AboutSection, 
        "features": FeaturesList
    }

    return Agent(
        name=f"{content_type.title()}ValidationAgent",
        model="gpt-4o-mini",
        instructions=validation_instructions,
        output_type=output_type_map.get(content_type, dict),
        model_settings=ModelSettings(temperature=0.1)
    )
