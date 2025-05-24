"""
Database utilities for Firestore connection
"""
import logging
import os
from google.cloud import firestore
from collections import defaultdict

logger = logging.getLogger(__name__)

# Mock database for local development
class MockFirestore:
    """A simple mock of Firestore for local development when credentials aren't available."""
    
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(dict))
        logger.info("Using MockFirestore for local development")
    
    def collection(self, collection_name):
        return MockCollection(self, collection_name)

class MockCollection:
    def __init__(self, db, collection_name):
        self.db = db
        self.collection_name = collection_name
    
    def document(self, doc_id=None):
        # Generate a random doc_id if none is provided
        if doc_id is None:
            import uuid
            doc_id = str(uuid.uuid4())
        return MockDocument(self.db, self.collection_name, doc_id)
    
    def where(self, field, op, value):
        # Simple implementation that doesn't actually filter
        return self
    
    def stream(self):
        # Return empty list for now
        return []

class MockDocument:
    def __init__(self, db, collection_name, doc_id):
        self.db = db
        self.collection_name = collection_name
        self.doc_id = doc_id
        
    def collection(self, subcollection_name):
        return MockCollection(self.db, f"{self.collection_name}/{self.doc_id}/{subcollection_name}")
    
    def set(self, data, merge=False):
        self.db.data[self.collection_name][self.doc_id] = data
        return True
    
    def update(self, data):
        if self.doc_id in self.db.data[self.collection_name]:
            self.db.data[self.collection_name][self.doc_id].update(data)
        else:
            self.db.data[self.collection_name][self.doc_id] = data
        return True
    
    def get(self):
        return MockDocumentSnapshot(self.db.data[self.collection_name].get(self.doc_id, {}))

class MockDocumentSnapshot:
    def __init__(self, data):
        self.data = data
        self.exists = bool(data)
    
    def to_dict(self):
        return self.data

# Global mock instance
_mock_db = None

def get_db():
    """Initialize and return Firestore client or mock for local development."""
    global _mock_db
    
    # Check if we're in local development mode
    use_mock = os.environ.get("USE_MOCK_DB", "false").lower() == "true"
    
    try:
        if use_mock:
            if _mock_db is None:
                _mock_db = MockFirestore()
            return _mock_db
        else:
            return firestore.Client()
    except Exception as e:
        logger.warning(f"Failed to initialize Firestore client: {e}. Using MockFirestore instead.")
        if _mock_db is None:
            _mock_db = MockFirestore()
        return _mock_db
