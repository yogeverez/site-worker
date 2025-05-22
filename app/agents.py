from agents import Agent, Runner, function_tool, ModelSettings
from schemas import HeroSection, AboutSection, FeaturesList

# Define the Hero section agent
hero_agent = Agent(
    name="HeroSectionAgent",
    instructions=(
        "You are a copywriter for personal websites. Use the user's information to create a JSON for the hero section. "
        "The hero JSON should include the person's name as a bold headline and a one-sentence tagline highlighting their role or uniqueness. "
        "Output only valid JSON for the HeroSection model."
    ),
    model="gpt-4-0613",  # GPT-4 chat model with function-calling (0613 or latest)
    output_type=HeroSection  # Pydantic model for validation
)

# Define the About section agent
about_agent = Agent(
    name="AboutSectionAgent",
    instructions=(
        "You write an 'About Me' section for a personal site in third person. Summarize the user's background, skills, and interests in a single paragraph. "
        "Output only valid JSON for the AboutSection model."
    ),
    model="gpt-4-0613",
    output_type=AboutSection
)

# Define the Features list section agent (e.g., key achievements or offerings)
features_agent = Agent(
    name="FeaturesListAgent",
    instructions=(
        "You create a features/skills list section for a personal site. Pick 3 to 5 key points about the person (achievements, skills, or services) and output them as a list. "
        "Each feature has a short title and a one-sentence description. "
        "Output only valid JSON for the FeaturesList model."
    ),
    model="gpt-4-0613",
    output_type=FeaturesList
)

# (Optional) Define a Researcher agent if using agent-based web search (not used in this implementation)
# researcher_agent = Agent(
#     name="ResearchAgent",
#     instructions="Find relevant online information about the user given their name, title, and social profiles.",
#     tools=[ ... ]  # Could include a web search tool, etc.
# )

# Translator function tool to translate text to a target language
@function_tool
def translate_text(text: str, target_language: str) -> str:
    """Translate the given text into the target language while preserving meaning."""
    import openai
    # Use a smaller model for translation for efficiency
    translate_prompt = f"Translate the following text to {target_language}:\n```{text}```"
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "system", "content": "You are a translation assistant."},
                {"role": "user", "content": translate_prompt}
            ],
            temperature=0.3
        )
        translated = resp["choices"][0]["message"]["content"].strip()
        return translated
    except Exception as e:
        # If translation fails, return original text as fallback
        return text
