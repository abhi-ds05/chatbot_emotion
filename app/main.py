from fastapi import FastAPI
from app.routes import router  # Import API routes

# Function to create and configure FastAPI app
def create_app() -> FastAPI:
    app = FastAPI(title="Emotion-Aware Chatbot")

    # Include your router (from routes.py)
    app.include_router(router)

    # TODO: Add middleware (e.g., logging, CORS, error tracking) if needed

    return app

# Create the app instance
app = create_app()

# Optional: Run the app if this file is the script entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
