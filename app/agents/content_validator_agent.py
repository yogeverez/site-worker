# app/agents/content_validator_agent.py

from agents import Agent, ModelSettings, AgentOutputSchema
from app.schemas import HeroSection, AboutSection, FeaturesList

def content_validator_agent(content_type: str) -> Agent:
    """
    Factory for an agent that validates and corrects a given content section.
    content_type should be one of: "hero", "about", "features".
    """
    instructions = f"""You are a validator for the {content_type} section content.
Check the given {content_type} section content for:
- Factual accuracy (ensure claims are supported by provided info)
- Professional tone and appropriate style
- Correct JSON structure for the {content_type.capitalize()}Section model
- Completeness of required fields.

If issues are found:
- Correct minor errors directly (e.g., fix JSON formatting, adjust tone or remove unsupported claims).
- If content is fundamentally flawed or unsupported by facts, recommend regeneration.

Output the validated {content_type} section JSON. If it was valid, you may output it unchanged.
"""
    # Choose the Pydantic model corresponding to the content_type
    output_model = {"hero": HeroSection, "about": AboutSection, "features": FeaturesList}.get(content_type, dict)
    return Agent(
        name=f"{content_type.capitalize()}ValidationAgent",
        instructions=instructions,
        model="gpt-4o-mini",
        output_type=AgentOutputSchema(output_model, strict_json_schema=False),
        model_settings=ModelSettings(temperature=0.1, max_tokens=400)
    )
