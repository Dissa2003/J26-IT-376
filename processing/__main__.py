import uvicorn

uvicorn.run("processing.app:app", host="0.0.0.0", port=8002)