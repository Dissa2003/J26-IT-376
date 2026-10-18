"""Run the standalone policy verification API."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("api.routes:app", host="0.0.0.0", port=8002)
