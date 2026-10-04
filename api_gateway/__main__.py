import uvicorn

uvicorn.run("api_gateway.app:app", host="0.0.0.0", port=8000)