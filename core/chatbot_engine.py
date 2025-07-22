# core/chatbot_engine.py

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import List, Optional

class ChatbotEngine:
    """
    Emotion-aware chatbot engine that uses an open-source LLM.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_context_length: int = 2048,
        device: Optional[str] = None
    ):
        """
        Initialize the chatbot engine with a Hugging Face LLM pipeline.

        Args:
            model_name (str): The name of the local or remote Hugging Face model.
            max_context_length (int): Max tokens to include in prompt.
            device (str or None): 'cuda', 'cpu', or auto-detected.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.max_context_length = max_context_length

    def build_prompt(self, chat_history: List[str], emotion: Optional[str] = None) -> str:
        """
        Build a conversational prompt including optional emotion context.

        Args:
            chat_history (List[str]): History of turns (user + bot).
            emotion (str): Optional emotion tag to guide the model.

        Returns:
            str: The final prompt string.
        """
        preamble = (
            f"The user is feeling {emotion}.\n"
            if emotion else ""
        )
        chat_transcript = "\n".join(chat_history[-10:])  # Keep last 10 turns
        return f"{preamble}Continue the conversation below:\n{chat_transcript}\nBot:"

    def generate_response(self, chat_history: List[str], emotion: Optional[str] = None) -> str:
        """
        Generate a response using the LLM.

        Args:
            chat_history (List[str]): User + bot messages so far.
            emotion (str): Emotion category to condition response.

        Returns:
            str: Model-generated bot reply.
        """
        prompt = self.build_prompt(chat_history, emotion)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_context_length)
        input_ids = inputs["input_ids"].to(self.model.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the generated part after the last 'Bot:' marker
        if "Bot:" in output_text:
            return output_text.split("Bot:")[-1].strip()
        else:
            return output_text.strip()

