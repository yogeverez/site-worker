from pydantic import BaseModel, Field
from typing import List

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
