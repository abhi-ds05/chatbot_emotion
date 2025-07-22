# app/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.response_generator import ResponseGenerator

router = APIRouter()

# Request and response schemas
class ChatRequest(BaseModel):
    user_id: str
    message: str
    emotion: str = None  # Optional, can be detected internally

class ChatResponse(BaseModel):
    response: str
    context_emotion: str
    chat_history: list

# Initialize your core chatbot engine instance once per app
response_generator = ResponseGenerator()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint to receive user message and return chatbot response.
    """
    try:
        result = response_generator.generate_response(
            user_id=request.user_id,
            user_message=request.message,
            emotion=request.emotion
        )
        return {
            "response": result["response"],
            "context_emotion": result.get("context_emotion", "neutral"),
            "chat_history": result.get("chat_history", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
