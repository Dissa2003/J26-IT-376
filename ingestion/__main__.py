"""Run the ingestion service with Uvicorn."""

import uvicorn

uvicorn.run("ingestion.app:app", host="0.0.0.0", port=8001)