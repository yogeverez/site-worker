"""
Translator Agent - Localizes content to different languages
"""
import logging
from app.agent_types import Agent, ModelSettings

logger = logging.getLogger(__name__)

def translator_agent() -> Agent:
    """
    Translation agent for localizing content to different languages.
    """
    instructions = """You are a professional translator specializing in website content localization.
    
    Your responsibilities:
    - Translate content accurately while preserving meaning and tone
    - Adapt content for cultural nuances when appropriate
    - Maintain professional terminology consistency
    - Ensure translations flow naturally in the target language
    
    Always provide high-quality translations that read naturally to native speakers."""

    return Agent(
        name="TranslatorAgent",
        instructions=instructions,
        model="gpt-4o-mini",
        model_settings=ModelSettings(
            temperature=0.2,
            max_tokens=3000
        )
    )

def translate_text(text: str, target_language: str) -> str:
    """
    Translate text to the specified target language using the translator agent.
    
    Args:
        text: Text to translate
        target_language: Target language (e.g., 'Spanish', 'French', 'German')
        
    Returns:
        Translated text
    """
    try:
        agent = translator_agent()
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        
        runner = Runner()
        result = runner.run(agent, prompt, max_turns=1)
        
        if result and hasattr(result, 'final_output'):
            return result.final_output
        return text  # Return original if translation fails
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text  # Return original text if translation fails
