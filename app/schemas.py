# app/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


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


class ResearchDoc(BaseModel):
    title: str = Field(..., description="Title of the research document or webpage")
    url: str = Field(..., description="URL of the source")
    snippet: Optional[str] = Field(None, description="Short snippet or preview of the content")

    # We only store the truncated summary in Firestore,
    # so we don’t need fields like raw_content or translations here.
    content: str = Field(..., description="Summarized content from the source")


class UserProfileData(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the person")
    current_title: Optional[str] = Field(None, description="Current job title or professional role")
    current_company: Optional[str] = Field(None, description="Current company or organization")
    bio: Optional[str] = Field(None, description="Personal or professional bio")
    skills: List[str] = Field(default_factory=list, description="List of professional skills")
    education: List[str] = Field(default_factory=list, description="Educational background")
    achievements: List[str] = Field(default_factory=list, description="Notable achievements or awards")
    projects: List[str] = Field(default_factory=list, description="Notable projects or works")
    social_profiles: Dict[str, str] = Field(default_factory=dict, description="Social media and professional profile URLs")
    languages_spoken: List[str] = Field(default_factory=list, description="Languages the person speaks")
    location: Optional[str] = Field(None, description="Current location or residence")

    class Config:
        extra = "allow"  # In case we want to pass through minor unexpected fields
