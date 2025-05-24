# app/content_generation_agent.py  (refactored helpers only)

import json
from app.database import get_db

def get_research_facts(uid: str, session_id: int) -> str:
    """
    Compile key facts from research documents for the given user (from Firestore).
    Returns a JSON string (list of fact snippets).
    """
    db = get_db()
    sources_ref = db.collection("research").document(uid).collection("sources")
    docs = sources_ref.stream()
    facts = []
    try:
        for i, doc in enumerate(docs):
            if i >= 5:  # limit to first 5 sources to constrain size
                break
            data = doc.to_dict()
            snippet = data.get("snippet") or ""
            if snippet:
                facts.append(snippet)
    except Exception as e:
        return json.dumps([])  # return empty list on error
    # Return facts as a JSON array string
    return json.dumps(facts, ensure_ascii=False)
 
def save_content_section(uid: str, section_type: str, content: str, language: str = "en") -> bool:
    """
    Save a generated content section (already JSON) to Firestore under content/{uid}/{language}/{section_type}.
    """
    db = get_db()
    try:
        content_data = json.loads(content) if isinstance(content, str) else content
    except Exception:
        content_data = content  # if content is already a dict or not valid JSON string
    doc_ref = db.collection("content").document(uid).collection(language).document(section_type)
    try:
        doc_ref.set(content_data)
        return True
    except Exception as e:
        logger = __import__('logging').getLogger(__name__)
        logger.error(f"Failed to save content section {section_type}: {e}")
        return False
