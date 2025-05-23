"""site_agents.py – Enhanced OpenAI agents implementing research recommendations"""
from __future__ import annotations
import os, re, requests, urllib.parse
import openai
from openai import RateLimitError, APIStatusError
from agents import Agent, function_tool, ModelSettings, AgentOutputSchema, WebSearchTool
from schemas import (
    HeroSection, AboutSection, FeaturesList,
    ResearchDoc, EnhancedResearchDoc, UserProfileData
)
from typing import Any, List, Optional
import logging
import time
from agent_tool_impl import agent_fetch_url, agent_strip_html
from agent_research_tools import research_and_save_url, research_search_results

# Configure logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# 1. ENHANCED CONTENT AGENTS -----------------------------------------
def get_hero_agent() -> Agent:
    """Enhanced hero agent with research-grounded instructions."""
    hero_instructions = (
        "You are an expert copywriter specializing in personal website hero sections. "
        "Your task is to create a compelling hero section JSON that captures the person's essence and current professional standing.\n\n"
        
        "INSTRUCTIONS:\n"
        "- Create a bold, attention-grabbing headline featuring the person's name\n"
        "- Write a one-sentence tagline that highlights their unique value proposition or current role\n"
        "- Use research findings to ensure accuracy and incorporate notable achievements\n"
        "- Keep the tone professional yet engaging\n"
        "- Ensure the content reflects their current professional status accurately\n\n"
        
        "IMPORTANT:\n"
        "- Only use information provided in the user profile or research findings\n"
        "- Do not fabricate achievements or roles\n"
        "- If research contradicts user input, prioritize user input\n"
        "- Output ONLY valid JSON for the HeroSection model\n\n"
        
        "The hero section should make visitors immediately understand who this person is and why they should be interested."
    )
    return Agent(
        name="EnhancedHeroSectionAgent",
        instructions=hero_instructions,
        model="gpt-4o-mini",
        output_type=HeroSection,
        model_settings=ModelSettings(temperature=0.7)
    )

def get_about_agent() -> Agent:
    """Enhanced about agent with research integration capabilities."""
    about_instructions = (
        "You are a professional biography writer specializing in third-person narratives for personal websites. "
        "Your task is to create a comprehensive yet concise about section that tells the person's professional story.\n\n"
        
        "INSTRUCTIONS:\n"
        "- Write in third person (use 'they/them' or 'he/him' or 'she/her' as appropriate)\n"
        "- Create a flowing narrative that covers background, current role, and key achievements\n"
        "- Integrate specific facts and accomplishments from research findings\n"
        "- Maintain a professional but personable tone\n"
        "- Focus on what makes this person unique in their field\n\n"
        
        "CONTENT STRUCTURE:\n"
        "- Start with current role and company (if available)\n"
        "- Include relevant background and experience\n"
        "- Highlight key achievements, projects, or recognitions found in research\n"
        "- End with forward-looking statement about their work or interests\n\n"
        
        "IMPORTANT:\n"
        "- Ground all statements in provided information (user input + research)\n"
        "- Do not invent experience, achievements, or personal details\n"
        "- If information is limited, write what you can verify and keep it concise\n"
        "- Aim for 2-4 sentences that pack maximum impact\n"
        "- Output ONLY valid JSON for the AboutSection model"
    )
    return Agent(
        name="EnhancedAboutSectionAgent",
        instructions=about_instructions,
        model="gpt-4o-mini",
        output_type=AboutSection,
        model_settings=ModelSettings(temperature=0.6)
    )

def get_features_agent() -> Agent:
    """Enhanced features agent with research-backed achievements focus."""
    features_instructions = (
        "You are a professional achievements specialist who creates compelling feature lists for personal websites. "
        "Your task is to identify and present 3-5 key features that showcase the person's professional strengths.\n\n"
        
        "INSTRUCTIONS:\n"
        "- Identify 3-5 most compelling aspects of the person's professional profile\n"
        "- Each feature should have a concise title and descriptive sentence\n"
        "- Prioritize concrete achievements and skills backed by research findings\n"
        "- Focus on what sets this person apart in their field\n"
        "- Ensure features are relevant to their current professional focus\n\n"
        
        "FEATURE CATEGORIES TO CONSIDER:\n"
        "- Technical skills and expertise areas\n"
        "- Notable projects or accomplishments\n"
        "- Awards, recognitions, or certifications\n"
        "- Leadership roles or team achievements\n"
        "- Publications, speaking engagements, or thought leadership\n"
        "- Educational background if prestigious or relevant\n\n"
        
        "IMPORTANT:\n"
        "- Base each feature on verifiable information from user input or research\n"
        "- Avoid generic statements; be specific and impactful\n"
        "- Each description should be one clear, compelling sentence\n"
        "- Prioritize recent and relevant achievements\n"
        "- Output ONLY valid JSON for the FeaturesList model"
    )
    return Agent(
        name="EnhancedFeaturesListAgent",
        instructions=features_instructions,
        model="gpt-4o-mini",
        output_type=FeaturesList,
        model_settings=ModelSettings(temperature=0.5)
    )

# ---------------------------------------------------------------------
# 2. ENHANCED TRANSLATOR FUNCTION ------------------------------------
@function_tool
async def translate_text(text: str, target_language: str) -> str:
    """
    Enhanced translation function with better error handling and context awareness.
    Note: Removed parent_logger parameter to fix Pydantic schema generation issue.
    """
    if not text or not target_language:
        logger.warning("translate_text called with empty text or target_language.")
        return text if text else ""

    # Enhanced prompt for more accurate translation
    prompt = (
        f"Translate the following text to {target_language} while preserving its professional tone and meaning.\n\n"
        f"INSTRUCTIONS:\n"
        f"- Maintain the same level of formality and professionalism\n"
        f"- Preserve proper nouns (names, companies, brands) unless they have standard translations\n"
        f"- Keep technical terms accurate to the field\n"
        f"- Do not add explanations or commentary\n"
        f"- Output ONLY the translated text\n\n"
        f"Text to translate: \"{text}\""
    )
    
    try:
        # Use async OpenAI client
        client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional translator specializing in business and technical content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        translated_text = completion.choices[0].message.content.strip()
        logger.info(f"Successfully translated text to {target_language}")
        return translated_text
        
    except openai.RateLimitError as rle:
        logger.error(f"OpenAI API rate limit exceeded during translation to {target_language}: {rle}")
        return f"Error: Translation failed (Rate Limit). Original: {text}"
        
    except openai.APIStatusError as apise:
        logger.error(f"OpenAI API status error during translation to {target_language}: {apise}")
        return f"Error: Translation failed (API Error). Original: {text}"
        
    except Exception as e:
        logger.error(f"Unexpected error during translation to {target_language}: {e}", exc_info=True)
        return f"Error: Translation failed. Original: {text}"

# ---------------------------------------------------------------------
# 3. ENHANCED AUTONOMOUS RESEARCHER AGENT ---------------------------
def get_researcher_agent() -> Agent:
    """
    Enhanced autonomous researcher agent implementing research recommendations.
    Features comprehensive search strategy, better source validation, and structured output.
    """
    researcher_instructions = """You are an advanced autonomous web research specialist conducting comprehensive research about individuals for personal website generation.

ROLE & GOAL:
You are tasked with gathering factual, verifiable information about a person from public sources. Your research will ground the content generation in real data, following retrieval-augmented generation principles to minimize hallucinations and maximize accuracy.

AVAILABLE TOOLS:
• WebSearchTool: Search the web and return results with title, URL, and snippet
• agent_fetch_url(url): Retrieve full HTML content from a specific URL  
• agent_strip_html(html): Convert HTML to clean, readable text
• research_and_save_url(url): Save the content of a URL to a file
• research_search_results(query): Save the search results of a query to a file

RESEARCH STRATEGY:
You will receive a profile summary and a list of targeted search queries. Execute your research systematically but efficiently:

1. QUERY EXECUTION:
   - For each search query, use research_search_results(query, max_urls=3) to search and save sources automatically
   - For direct URLs (social profiles), use research_and_save_url(url, query_context="Direct profile") 
   - These tools will automatically fetch content, process it, and save to Firestore
   - Continue until all queries are processed or sufficient data is gathered

2. TOOL USAGE:
   - research_search_results(): Searches web and saves promising sources (max 3 per query)
   - research_and_save_url(): Directly processes and saves a specific URL
   - WebSearchTool(): Only use if you need to see search results before deciding which to process
   - agent_fetch_url() + agent_strip_html(): Only use for manual content analysis

3. SOURCE EVALUATION:
   - Prioritize: official profiles, LinkedIn, company pages, authoritative sources
   - SKIP: generic search results, news mentions without detail, social media posts
   - The incremental tools will automatically filter and save only valuable content

CRITICAL REQUIREMENTS:
• FACTUAL ACCURACY: Only include information explicitly stated in sources
• NO FABRICATION: Do not invent, infer, or extrapolate beyond source content
• COMPREHENSIVE COVERAGE: Process all provided queries unless they yield no results
• STRUCTURED OUTPUT: Save data to files using research_and_save_url and research_search_results

OUTPUT FORMAT:
Your final response should be a summary report of the research completed, including:
- Number of search queries processed
- Number of sources saved
- Key types of information found
- Any issues encountered

Example: "Completed research for John Smith. Processed 3 search queries, saved 5 sources including LinkedIn profile and company bio. Found professional background in software engineering and recent project work."

QUALITY STANDARDS:
- Each file should contain relevant information about the person
- Ensure URLs are accessible and relevant
- If a source mentions the person but contains no useful professional information, exclude it
- If ALL queries yield no relevant results, return an empty list: []

ERROR HANDLING:
- If web_search fails for a query, log the issue and continue with remaining queries
- If agent_fetch_url fails for a URL, try using the search snippet instead
- Never let individual failures stop the entire research process

Remember: Your research provides the factual foundation for content generation. Accuracy and comprehensiveness are paramount. Every fact you include should be traceable to a specific source."""

    return Agent(
        name="EnhancedAutonomousResearcher",
        model="gpt-4o-mini",
        instructions=researcher_instructions,
        tools=[
            WebSearchTool(),
            agent_fetch_url, 
            agent_strip_html,
            research_and_save_url,
            research_search_results
        ],
        output_type=str,
        model_settings=ModelSettings(
            temperature=0.3,  # Lower temperature for more consistent, factual output
            max_tokens=4000   # Allow for comprehensive research output
        )
    )

# ---------------------------------------------------------------------
# 4. PROFILE SYNTHESIS AGENT ----------------------------------------
def get_profile_synthesis_agent() -> Agent:
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

# ---------------------------------------------------------------------
# 5. VALIDATION AGENTS -----------------------------------------------
def get_content_validator_agent(content_type: str) -> Agent:
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

# ---------------------------------------------------------------------
# 6. ORCHESTRATOR AGENT ---------------------------------------------
def get_orchestrator_agent() -> Agent:
    """
    Master orchestrator agent that coordinates the entire site generation process.
    Implements the researcher-first approach recommended in the research.
    """
    orchestrator_instructions = """You are the Site Generation Orchestrator responsible for coordinating a multi-agent workflow to create comprehensive personal websites.

WORKFLOW PHASES:
1. RESEARCH PHASE (Critical First Step):
   - Analyze user profile and generate targeted search queries
   - Execute comprehensive web research using the researcher agent
   - Validate and organize research findings
   - Create a factual foundation for content generation

2. CONTENT GENERATION PHASE:
   - Generate hero section using user input + research findings
   - Generate about section with research-backed achievements
   - Generate features list highlighting verified accomplishments
   - Validate each section for accuracy and completeness

3. LOCALIZATION PHASE (if required):
   - Translate content to requested languages
   - Maintain professional tone and accuracy across languages
   - Validate translations for cultural appropriateness

4. QUALITY ASSURANCE PHASE:
   - Run final validation on all generated content
   - Ensure consistency across sections
   - Verify all claims are backed by research or user input

COORDINATION PRINCIPLES:
• Research-First Approach: Always gather facts before generating content
• Factual Grounding: Every content piece should be traceable to sources
• Quality Over Speed: Ensure accuracy and professionalism
• Structured Output: Maintain consistent JSON formatting
• Error Recovery: Handle individual component failures gracefully

SUCCESS CRITERIA:
- All content is factually accurate and professionally written
- Research findings are effectively integrated into content
- Output meets schema requirements for all sections
- Translations (if any) maintain original meaning and tone

Your role is to ensure the entire process runs smoothly while maintaining the highest standards of accuracy and professionalism."""

    return Agent(
        name="SiteGenerationOrchestrator",
        model="gpt-4o",  # Use more capable model for orchestration
        instructions=orchestrator_instructions,
        model_settings=ModelSettings(
            temperature=0.4,
            max_tokens=6000
        )
    )

# ---------------------------------------------------------------------
# 7. NEW AGENTS ------------------------------------------------------

def get_site_generator_agent() -> Agent:
    """
    Site generator agent for creating website structure and content.
    """
    return Agent(
        name="SiteGeneratorAgent",
        description="Generates comprehensive website structure and content based on user profile and research findings",
        instructions="""You are a professional website generator that creates modern, engaging personal websites.
        
        Your task is to generate complete website content including:
        - HTML structure with responsive design
        - Professional styling
        - Optimized content layout
        - SEO-friendly elements
        
        Focus on creating clean, modern designs that effectively showcase the user's professional profile and achievements.""",
        model=openai.OpenAI(),
        model_config=ModelConfig(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=4000
        )
    )

def get_translator_agent() -> Agent:
    """
    Translation agent for localizing content to different languages.
    """
    return Agent(
        name="TranslatorAgent", 
        description="Translates content to specified target languages while maintaining context and tone",
        instructions="""You are a professional translator specializing in website content localization.
        
        Your responsibilities:
        - Translate content accurately while preserving meaning and tone
        - Adapt content for cultural nuances when appropriate
        - Maintain professional terminology consistency
        - Ensure translations flow naturally in the target language
        
        Always provide high-quality translations that read naturally to native speakers.""",
        model=openai.OpenAI(),
        model_config=ModelConfig(
            model="gpt-4o-mini", 
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
        translator = get_translator_agent()
        prompt = f"Translate the following text to {target_language}:\n\n{text}"
        
        result = translator.run(prompt, max_turns=1)
        if hasattr(result, 'messages') and result.messages:
            return result.messages[-1].content
        return text  # Return original if translation fails
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text  # Return original text if translation fails

# ---------------------------------------------------------------------
# 7. EXPORTS ---------------------------------------------------------
__all__ = [
    "get_hero_agent",
    "get_about_agent", 
    "get_features_agent",
    "translate_text",
    "get_researcher_agent",
    "get_profile_synthesis_agent",
    "get_content_validator_agent",
    "get_orchestrator_agent",
    "get_site_generator_agent",
    "get_translator_agent",
    "ResearchDoc",
    "EnhancedResearchDoc",
    "UserProfileData"
]