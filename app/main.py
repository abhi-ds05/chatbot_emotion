# app/main.py

from fastapi import FastAPI
from app.routes import router

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI app.
    """
    app = FastAPI(
        title="Emotion-Aware Multimodal Chatbot",
        description="A chatbot capable of detecting emotion from text, audio, and images.",
        version="1.0.0"
    )
    # Register your API routes from routes.py
    app.include_router(router)
    # If you use custom middleware, import and add here (optional)
    return app

# Instantiate the app for ASGI servers (uvicorn, etc.)
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
