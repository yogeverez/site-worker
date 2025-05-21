"""
Pydantic models that mirror the client-side Zod schemas.
Guardrails will validate LLM output against these.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class HeroComponent(BaseModel):
    title: str
    subtitle: str
    ctaText: str
    backgroundImageUrl: Optional[str] = None

class AboutComponent(BaseModel):
    sectionTitle: str
    bio: str
    imageUrl: Optional[str] = None

class FeatureItem(BaseModel):
    title: str
    description: str
    icon: Optional[str] = None

class FeaturesListComponent(BaseModel):
    sectionTitle: str
    features: List[FeatureItem]

class PortfolioItem(BaseModel):
    title: str
    description: str
    imageUrl: Optional[str] = None
    projectUrl: Optional[str] = None

class PortfolioGridComponent(BaseModel):
    sectionTitle: str
    items: List[PortfolioItem]

class TimelineItem(BaseModel):
    date: str
    title: str
    subtitle: Optional[str] = None
    description: Optional[str] = None

class ExperienceTimelineComponent(BaseModel):
    sectionTitle: str
    items: List[TimelineItem]

class TestimonialItem(BaseModel):
    quote: str
    author: str
    role: Optional[str] = None
    authorImageUrl: Optional[str] = None

class TestimonialsComponent(BaseModel):
    sectionTitle: str
    testimonials: List[TestimonialItem]

class SocialLink(BaseModel):
    platform: str
    url: str

class ContactComponent(BaseModel):
    sectionTitle: str
    blurb: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    socialLinks: List[SocialLink]

class NewsletterSignupComponent(BaseModel):
    sectionTitle: str
    description: str
    ctaLabel: str

class ScheduleEvent(BaseModel):
    time: str
    title: str
    speaker: Optional[str] = None
    description: Optional[str] = None

class ScheduleComponent(BaseModel):
    sectionTitle: str
    events: List[ScheduleEvent]
