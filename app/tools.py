"""
Shared utility functions (web fetch, search, image search, Firestore write).
"""
from __future__ import annotations
import os, random, re, requests
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firestore client
try:
    # In Cloud Run, this will use the default service account credentials
    # In local development, it will use Application Default Credentials if available
    # or raise an exception which we'll catch
    from google.cloud import firestore
    db = firestore.Client()
    FIRESTORE_AVAILABLE = True
    logger.info("Firestore client initialized successfully")
except Exception as e:
    logger.warning(f"Firestore client initialization failed: {e}")
    logger.warning("Running in local development mode without Firestore")
    db = None
    FIRESTORE_AVAILABLE = False

# ------------------------------------------------------------------ #
# 1.  Basic web fetching / stripping
# ------------------------------------------------------------------ #
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SiteGeneratorBot/1.0)"}
TAG_RE = re.compile(r"<[^>]+>")

def fetch_url(url: str) -> str:
    """Return raw HTML (limited to 100 kB)."""
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text[:100_000]

def strip_html(html: str) -> str:
    """Very naive HTML → plaintext."""
    return TAG_RE.sub(" ", html)

# ------------------------------------------------------------------ #
# 2.  Web & image search
# ------------------------------------------------------------------ #
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERP_ENDPOINT = "https://serpapi.com/search.json"
BING_SEARCH_URL = os.getenv("BING_SEARCH_URL", "")  # optional
UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")
UNSPLASH_ENDPOINT = "https://api.unsplash.com/search/photos"

def web_search(query: str, k: int = 10) -> List[str]:
    """Return top-k result URLs for a query."""
    if SERPAPI_KEY:
        params = {"api_key": SERPAPI_KEY, "q": query, "num": k}
        try:
            resp = requests.get(SERP_ENDPOINT, params=params, timeout=15).json()
            return [r["link"] for r in resp.get("organic_results", [])][:k]
        except Exception as e:
            print("SerpAPI failed:", e)

    # fallback – crude Bing scrape
    if BING_SEARCH_URL:
        html = requests.get(BING_SEARCH_URL + query, headers=HEADERS, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
        return [a["href"] for a in soup.select("li.b_algo h2 a")[:k]]
    return []

def image_search(query: str, k: int = 5) -> List[str]:
    """Unsplash free-to-use images."""
    if not UNSPLASH_KEY:
        return []
    headers = {"Authorization": f"Client-ID {UNSPLASH_KEY}"}
    resp = requests.get(UNSPLASH_ENDPOINT, params={"query": query, "per_page": k}, headers=headers, timeout=15)
    resp.raise_for_status()
    return [p["urls"]["regular"] for p in resp.json().get("results", [])]

def random_image(query: str) -> str | None:
    imgs = image_search(query, 10)
    return random.choice(imgs) if imgs else None

# ------------------------------------------------------------------ #
# 3. Firestore writer
# ------------------------------------------------------------------ #
def save_component(uid: str, lang: str, component: str, data: Dict[str, Any]):
    """Write `{component}.json` into users/{uid}/siteContent/{lang}/components/."""
    if not FIRESTORE_AVAILABLE:
        logger.info(f"[LOCAL DEV] Would save component '{component}' for user {uid} in language {lang}")
        logger.info(f"[LOCAL DEV] Component data: {data}")
        return
        
    # Only execute if Firestore is available
    db.collection("users").document(uid) \
      .collection("siteContent").document(lang) \
      .collection("components").document(component).set(data)
