from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import datetime

# Original content schemas
class HeroSection(BaseModel):
    headline: str = Field(..., description="Main bold headline (usually the user's name or branding)")
    subheadline: str = Field(..., description="One sentence tagline or subtitle for the hero section")

class AboutSection(BaseModel):
    content: str = Field(..., description="A short paragraph in third person describing the user")

class FeatureItem(BaseModel):
    title: str = Field(..., description="Title of the feature/skill/achievement")
    description: str = Field(..., description="One sentence description or detail")

class FeaturesList(BaseModel):
    features: List[FeatureItem] = Field(..., description="List of key features or achievements about the user")

# Enhanced research schemas based on research recommendations
class ResearchDoc(BaseModel):
    title: str = Field(..., description="Title of the research document or webpage")
    url: str = Field(..., description="URL of the source")
    content: str = Field(..., description="Summarized content from the source")
    source_type: Optional[str] = Field(None, description="Type of source (e.g., 'linkedin', 'github', 'news_article', 'blog_post', 'company_website')")
    content_translations: Dict[str, str] = Field(default_factory=dict, description="Translations of content for different languages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the source")
    
    class Config:
        extra = "allow"  # Allow additional fields for future extensibility

# Enhanced user profile schema for better research targeting
class UserProfileData(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the person")
    current_title: Optional[str] = Field(None, description="Current job title or professional role")
    current_company: Optional[str] = Field(None, description="Current company or organization")
    bio: Optional[str] = Field(None, description="Personal/professional bio")
    skills: List[str] = Field(default_factory=list, description="List of professional skills")
    education: List[str] = Field(default_factory=list, description="Educational background")
    achievements: List[str] = Field(default_factory=list, description="Notable achievements or awards")
    projects: List[str] = Field(default_factory=list, description="Notable projects or works")
    social_profiles: Dict[str, str] = Field(default_factory=dict, description="Social media and professional profile URLs")
    languages_spoken: List[str] = Field(default_factory=list, description="Languages the person speaks")
    location: Optional[str] = Field(None, description="Current location or residence")
    
    class Config:
        extra = "allow"

# Research output schema that combines profile synthesis with source documentation
class ResearchOutput(BaseModel):
    profile: UserProfileData = Field(..., description="Synthesized profile data from research")
    sources: List[ResearchDoc] = Field(..., description="All research sources found")
    research_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the research process")
    
    class Config:
        extra = "allow"

# Enhanced manifest schema for comprehensive research tracking
class ResearchManifest(BaseModel):
    uid: str = Field(..., description="User ID for whom research was conducted")
    status: str = Field(..., description="Status of the research process")
    timestamp: datetime.datetime = Field(..., description="When the research was completed")
    
    # Query and search tracking
    search_queries_generated: List[str] = Field(default_factory=list, description="List of search queries used")
    direct_urls_identified: List[str] = Field(default_factory=list, description="Direct URLs to social profiles identified")
    successful_queries: List[str] = Field(default_factory=list, description="Queries that returned results")
    failed_queries: List[str] = Field(default_factory=list, description="Queries that failed or returned no results")
    
    # Results tracking
    total_sources_found: int = Field(0, description="Total number of sources discovered")
    sources_saved_successfully: int = Field(0, description="Number of sources successfully saved")
    saved_source_ids: List[str] = Field(default_factory=list, description="Firestore document IDs of saved sources")
    saved_source_urls: List[str] = Field(default_factory=list, description="URLs of all saved sources")
    source_types_found: List[str] = Field(default_factory=list, description="Types of sources discovered")
    
    # Quality and performance metrics
    research_duration_seconds: float = Field(0.0, description="Total time spent on research")
    average_content_length: float = Field(0.0, description="Average length of content found per source")
    unique_domains_found: int = Field(0, description="Number of unique domains in found sources")
    
    # Configuration and context
    template_type: Optional[str] = Field(None, description="Template type that influenced research strategy")
    max_queries_configured: int = Field(5, description="Maximum number of queries configured for this research")
    bypass_serpapi_enabled: bool = Field(False, description="Whether SerpAPI was bypassed due to limits")
    
    # Error tracking
    error_message: Optional[str] = Field(None, description="Error message if research failed")
    original_user_timestamp: Optional[int] = Field(None, description="Original timestamp from the user request")
    
    class Config:
        extra = "allow"

# Content generation metadata schema
class ContentGenerationMetadata(BaseModel):
    sections_attempted: int = Field(0, description="Number of content sections attempted")
    sections_completed: int = Field(0, description="Number of content sections successfully completed")
    research_facts_used: int = Field(0, description="Number of research facts available during generation")
    content_generation_errors: List[str] = Field(default_factory=list, description="List of errors encountered during generation")
    generation_duration_seconds: float = Field(0.0, description="Time spent generating content")
    research_data_available: bool = Field(False, description="Whether research data was available for content generation")
    template_type: Optional[str] = Field(None, description="Template type used for generation")
    generation_timestamp: Optional[datetime.datetime] = Field(None, description="When content generation occurred")
    
    class Config:
        extra = "allow"

# Translation metadata schema
class TranslationMetadata(BaseModel):
    languages_attempted: int = Field(0, description="Number of languages translation was attempted for")
    languages_completed: int = Field(0, description="Number of languages successfully translated")
    translation_errors: List[str] = Field(default_factory=list, description="List of translation errors encountered")
    translation_duration_seconds: float = Field(0.0, description="Time spent on translations")
    
    class Config:
        extra = "allow"

# Site generation status schema for comprehensive tracking
class SiteGenerationStatus(BaseModel):
    uid: str = Field(..., description="User ID")
    status: str = Field(..., description="Current status of site generation")
    last_updated: datetime.datetime = Field(..., description="Last update timestamp")
    
    # Research phase tracking
    research_completed: bool = Field(False, description="Whether research phase is completed")
    research_sources_count: int = Field(0, description="Number of research sources found")
    research_status: Optional[str] = Field(None, description="Status of research phase")
    
    # Content generation tracking
    content_generated: bool = Field(False, description="Whether content generation is completed")
    content_sections_completed: List[str] = Field(default_factory=list, description="List of content sections successfully generated")
    original_language: str = Field("en", description="Original language for content generation")
    
    # Translation tracking
    translations_completed: List[str] = Field(default_factory=list, description="List of languages for which translations are completed")
    translation_status: Optional[str] = Field(None, description="Status of translation phase")
    
    # Error tracking
    errors_encountered: List[str] = Field(default_factory=list, description="List of errors encountered during any phase")
    
    # Performance metrics
    total_processing_time_seconds: float = Field(0.0, description="Total time spent on all phases")
    
    class Config:
        extra = "allow"

# Query generation schema for better search strategy tracking
class SearchQueryStrategy(BaseModel):
    base_queries: List[str] = Field(default_factory=list, description="Basic queries based on name and role")
    template_specific_queries: List[str] = Field(default_factory=list, description="Queries specific to the template type")
    social_profile_queries: List[str] = Field(default_factory=list, description="Queries for finding social profiles")
    achievement_focused_queries: List[str] = Field(default_factory=list, description="Queries focused on finding achievements and accomplishments")
    direct_urls: List[str] = Field(default_factory=list, description="Direct URLs to fetch (e.g., provided social profiles)")
    
    template_type: str = Field("resume", description="Template type that influenced query strategy")
    max_queries: int = Field(5, description="Maximum number of queries to generate")
    
    class Config:
        extra = "allow"

# Source quality assessment schema
class SourceQualityMetrics(BaseModel):
    url: str = Field(..., description="URL of the source")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score (0-1)")
    credibility_score: float = Field(0.0, ge=0.0, le=1.0, description="Credibility score (0-1)")
    content_freshness: Optional[datetime.datetime] = Field(None, description="When the content was last updated")
    content_length: int = Field(0, description="Length of extracted content")
    domain_authority: Optional[str] = Field(None, description="Domain authority classification (high/medium/low)")
    
    # Quality indicators
    has_structured_data: bool = Field(False, description="Whether the source has structured data")
    is_primary_source: bool = Field(False, description="Whether this is a primary source (e.g., person's own profile)")
    is_verified_profile: bool = Field(False, description="Whether this is a verified social media profile")
    
    class Config:
        extra = "allow"

# Extended research doc with quality metrics
class EnhancedResearchDoc(ResearchDoc):
    quality_metrics: Optional[SourceQualityMetrics] = Field(None, description="Quality assessment of this source")
    extraction_method: str = Field("search", description="How this source was found (search/direct_url/recommendation)")
    processing_timestamp: Optional[datetime.datetime] = Field(None, description="When this source was processed")
    content_summary: Optional[str] = Field(None, description="AI-generated summary of the content")
    
    class Config:
        extra = "allow"