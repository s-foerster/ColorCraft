"""Runtime entrypoint for Google Cloud Run and other hosted environments."""

import os

import uvicorn


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
