# Site Worker

A Cloud Run worker service that uses OpenAI Agents SDK to generate website content based on user input.

## Overview

This service:
1. Listens for Pub/Sub messages containing user IDs and language preferences
2. Fetches site input data from Firestore
3. Uses OpenAI Agents to generate website content based on that input
4. Stores the generated components back in Firestore

## Project Structure

```
site-worker/
├── app/
│   ├── __init__.py
│   ├── main.py        # HTTP entry-point for Cloud Run
│   ├── tools.py       # Utility functions (web fetch, search, Firestore write)
│   ├── schemas.py     # Pydantic models for validation
│   ├── agents.py      # OpenAI Agents configuration
│   └── requirements.txt
├── Dockerfile
└── .github/
    └── workflows/
        └── deploy-cloud-run.yml
```

## Deployment

### GitHub Actions (Recommended)

This project includes a GitHub Actions workflow for automatic deployment to Cloud Run:

1. Push your code to the `main` branch or manually trigger the workflow
2. The workflow will:
   - Build the Docker image
   - Deploy to Cloud Run
   - Set up Pub/Sub topic and subscription (first deployment only)

Required GitHub secrets:
- `GCP_SA_KEY_DEV`: Service account key for development project
- `GCP_SA_KEY_PROD`: Service account key for production project
- `OPENAI_API_KEY`: OpenAI API key
- `SERPAPI_KEY`: SerpAPI key for web search
- `UNSPLASH_ACCESS_KEY`: Unsplash API key for image search

### Manual Deployment

```bash
# 1. Build container image
gcloud builds submit --tag gcr.io/$PROJECT_ID/site-worker

# 2. Deploy Cloud Run service
gcloud run deploy site-worker \
  --image gcr.io/$PROJECT_ID/site-worker \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --set-env-vars OPENAI_API_KEY=sk-***,SERPAPI_KEY=sa-***,UNSPLASH_ACCESS_KEY=un-***

# 3. Set up Pub/Sub
gcloud pubsub topics create site-generation-jobs
gcloud pubsub subscriptions create site-worker-sub \
  --topic site-generation-jobs \
  --push-endpoint=https://$(gcloud run services describe site-worker --region us-central1 --format='value(status.url)')/ \
  --push-auth-service-account=$PROJECT_ID@appspot.gserviceaccount.com
```

## Usage

To trigger site generation, publish a message to the `site-generation-jobs` topic with the following format:

```json
{
  "uid": "user123",
  "languages": ["en", "es", "fr"]
}
```

The service will generate website content for each language and store it in Firestore.

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI API access
- `SERPAPI_KEY`: Used for web search functionality
- `UNSPLASH_ACCESS_KEY`: Used for image search
- `BING_SEARCH_URL`: Optional fallback for web search

## Development

1. Install dependencies: `pip install -r app/requirements.txt`
2. Set environment variables
3. Run locally: `python app/main.py`
