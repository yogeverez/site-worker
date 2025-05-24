# app/agents/translator_agent.py

from typing import Dict, Any
from agents import Agent, Runner, ModelSettings

def translator_agent() -> Agent:
    """
    Agent for translating content to other languages.
    """
    instructions = """You are a professional translator for website content.
Translate any provided text into the target language accurately, preserving tone and meaning.
Adapt for cultural nuances as needed, and ensure the result reads naturally for a native speaker.
Output only the translated text."""
    return Agent(
        name="TranslatorAgent",
        instructions=instructions,
        model="gpt-4o-mini",
        model_settings=ModelSettings(temperature=0.2, max_tokens=1000)
    )

async def translate_text(text: str, target_language: str) -> str:
    """
    Helper function to translate text to the target language using the translator agent.
    
    Args:
        text: The text to translate
        target_language: The target language code (e.g., 'es', 'fr', 'he')
        
    Returns:
        The translated text
    """
    agent = translator_agent()
    runner = Runner()
    prompt = f"Translate the following text to {target_language}:\n\n{text}"
    result = await runner.run(agent, prompt)
    return result.final_output
