"""
Pydantic models that mirror the client-side Zod schemas.
Guardrails will validate LLM output against these.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Base model with configuration to disable additionalProperties
class StrictBaseModel(BaseModel):
    # The OpenAI Agents SDK requires a specific configuration for Pydantic models
    # We need to prevent additionalProperties from being included in the schema
    model_config = {
        "extra": "forbid",  # Forbid extra fields during validation
        "validate_assignment": True,
    }
    
    @classmethod
    def model_json_schema(cls, **kwargs):
        """Override the default schema generation to remove additionalProperties."""
        schema = super().model_json_schema(**kwargs)
        # Remove additionalProperties from the schema
        if "additionalProperties" in schema:
            del schema["additionalProperties"]
        return schema

class HeroComponent(StrictBaseModel):
    title: str
    subtitle: str
    ctaText: str
    backgroundImageUrl: Optional[str] = None

class AboutComponent(StrictBaseModel):
    sectionTitle: str
    bio: str
    imageUrl: Optional[str] = None

class FeatureItem(StrictBaseModel):
    title: str
    description: str
    icon: Optional[str] = None

class FeaturesListComponent(StrictBaseModel):
    sectionTitle: str
    features: List[FeatureItem]

class PortfolioItem(StrictBaseModel):
    title: str
    description: str
    imageUrl: Optional[str] = None
    projectUrl: Optional[str] = None

class PortfolioGridComponent(StrictBaseModel):
    sectionTitle: str
    items: List[PortfolioItem]

class TimelineItem(StrictBaseModel):
    date: str
    title: str
    subtitle: Optional[str] = None
    description: Optional[str] = None

class ExperienceTimelineComponent(StrictBaseModel):
    sectionTitle: str
    items: List[TimelineItem]

class TestimonialItem(StrictBaseModel):
    quote: str
    author: str
    role: Optional[str] = None
    authorImageUrl: Optional[str] = None

class TestimonialsComponent(StrictBaseModel):
    sectionTitle: str
    testimonials: List[TestimonialItem]

class SocialLink(StrictBaseModel):
    platform: str
    url: str

class ContactComponent(StrictBaseModel):
    sectionTitle: str
    blurb: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    socialLinks: List[SocialLink]

class NewsletterSignupComponent(StrictBaseModel):
    sectionTitle: str
    description: str
    ctaLabel: str

class ScheduleEvent(StrictBaseModel):
    time: str
    title: str
    speaker: Optional[str] = None
    description: Optional[str] = None

class ScheduleComponent(StrictBaseModel):
    sectionTitle: str
    events: List[ScheduleEvent]
