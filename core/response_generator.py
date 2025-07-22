# core/response_generator.py

from typing import Dict, Optional
from core.chatbot_engine import ChatbotEngine
from core.memory_manager import MemoryManager


class ResponseGenerator:
    """
    Orchestrates emotion-aware response generation by tying together
    memory, detected emotion, and the chatbot engine.
    """

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.engine = ChatbotEngine(model_name=model_name)
        self.memory = MemoryManager()

    def generate_response(self, user_id: str, user_message: str, emotion: Optional[str] = None) -> Dict[str, str]:
        """
        Generates a chatbot response using the user input, memory, and detected emotion.

        Args:
            user_id (str): Unique user identifier.
            user_message (str): The latest input message from the user.
            emotion (str, optional): The user's detected emotional state.

        Returns:
            dict: {
                "response": str,
                "context_emotion": Optional[str],
                "chat_history": List[Dict[str, str]]
            }
        """
        user_session = self.memory.get_user_memory(user_id)

        # Update session with latest user input and emotion
        user_session.add_message("user", user_message)
        if emotion:
            user_session.add_emotion(emotion)

        # Fetch recent message history for prompt building
        recent_history = [
            f"{entry['role'].capitalize()}: {entry['message']}"
            for entry in user_session.get_recent_history()
        ]

        # Generate response conditioned on detected emotion
        bot_reply = self.engine.generate_response(chat_history=recent_history, emotion=emotion or "neutral")

        # Update session with bot reply
        user_session.add_message("bot", bot_reply)

        # Return result
        return {
            "response": bot_reply,
            "context_emotion": emotion or user_session.get_last_emotion(),
            "chat_history": user_session.get_recent_history()
        }
