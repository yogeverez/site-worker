"""
Database utilities for Firestore connection
"""
import logging
from google.cloud import firestore

logger = logging.getLogger(__name__)

def get_db():
    """Initialize and return Firestore client."""
    try:
        return firestore.Client()
    except Exception as e:
        logger.error(f"Failed to initialize Firestore client: {e}")
        raise
