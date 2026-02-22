"""
LLM client configuration — MiniMax as the primary model.
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import MiniMaxChat

load_dotenv()


def get_llm(temperature: float = 0.0, model: str = "MiniMax-Text-01"):
    """Get a configured MiniMax LLM instance.

    Args:
        temperature: Sampling temperature (0.0 = deterministic).
        model: MiniMax model identifier.

    Returns:
        A MiniMaxChat instance ready to use.
    """
    api_key = os.getenv("MINIMAX_API_KEY")
    group_id = os.getenv("MINIMAX_GROUP_ID")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY not found. "
            "Set it in your .env file or environment variables."
        )
    if not group_id:
        raise ValueError(
            "MINIMAX_GROUP_ID not found. "
            "Set it in your .env file or environment variables."
        )

    return MiniMaxChat(
        model=model,
        temperature=temperature,
        minimax_api_key=api_key,
        minimax_group_id=group_id,
        max_tokens=4096,
    )


def call_llm(prompt: str, response_model=None, temperature: float = 0.0):
    """Call MiniMax with a prompt and optionally parse into a Pydantic model.

    Args:
        prompt: The prompt to send.
        response_model: Optional Pydantic model class for structured output.
        temperature: Sampling temperature.

    Returns:
        If response_model is provided, returns a parsed Pydantic object.
        Otherwise, returns the raw string response.
    """
    llm = get_llm(temperature=temperature)

    if response_model:
        structured_llm = llm.with_structured_output(response_model)
        return structured_llm.invoke(prompt)

    response = llm.invoke(prompt)
    return response.content
