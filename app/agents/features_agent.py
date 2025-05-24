# app/agents/features_agent.py

from agents import Agent, ModelSettings, AgentOutputSchema
from app.schemas import FeaturesList

def features_agent() -> Agent:
    """
    Agent for generating the Features/Skills section (list of highlights or skills).
    """
    instructions = (
        "You are a copywriter for professional highlights. Create a JSON FeaturesList of the user's key skills, accomplishments, or offerings.\n"
        "- Compile 3-5 bullet-point items, focusing on strengths and achievements.\n"
        "- Base each item on the user profile or research facts.\n"
        "- Ensure each feature is brief (one sentence) and impactful.\n"
        "- Output only valid JSON for the FeaturesList model."
    )
    return Agent(
        name="FeaturesAgent",
        instructions=instructions,
        model="gpt-4o-mini",
        output_type=AgentOutputSchema(FeaturesList, strict_json_schema=False),
        model_settings=ModelSettings(temperature=0.7, max_tokens=400)
    )
