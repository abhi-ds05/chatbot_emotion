# tests/test_chatbot_engine.py

import pytest
from core.chatbot_engine import ChatbotEngine

@pytest.fixture(scope="module")
def chatbot_engine():
    # Initialize the ChatbotEngine once per test module
    engine = ChatbotEngine()
    return engine

def test_generate_response(chatbot_engine):
    """
    Test that the chatbot engine generates a non-empty response given a valid prompt.
    """
    prompt = (
        "user: I am feeling happy today.\n"
        "bot:"
    )
    response = chatbot_engine.generate(prompt)
    assert isinstance(response, str), "Response should be a string"
    assert len(response.strip()) > 0, "Response should not be empty or whitespace"

def test_generate_empty_prompt(chatbot_engine):
    """
    Test how the chatbot engine handles an empty prompt.
    Response should still return a string or raise a controlled exception.
    """
    prompt = ""
    response = chatbot_engine.generate(prompt)
    assert isinstance(response, str), "Should return a string for empty prompt"

@pytest.mark.parametrize("invalid_input", [None, 123, 3.14, [], {}, True])
def test_generate_non_string_input(chatbot_engine, invalid_input):
    """
    Test input validation for non-string inputs.
    Expects a TypeError to be raised.
    """
    with pytest.raises(TypeError):
        chatbot_engine.generate(invalid_input)

def test_generate_whitespace_prompt(chatbot_engine):
    """
    Test how the chatbot engine handles a prompt that is only whitespace.
    """
    prompt = "    "
    response = chatbot_engine.generate(prompt)
    assert isinstance(response, str), "Should return a string for whitespace prompt"

def test_generate_special_characters(chatbot_engine):
    """
    Test response generation for prompt with special characters and unicode.
    """
    prompt = "user: 😊🤖💬! ¿Qué tal?\nbot:"
    response = chatbot_engine.generate(prompt)
    assert isinstance(response, str)
    assert len(response.strip()) > 0
