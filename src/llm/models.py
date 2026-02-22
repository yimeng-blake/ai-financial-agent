"""
LLM client configuration for Claude via Anthropic.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()


def get_llm(temperature: float = 0.0, model: str = "claude-sonnet-4-20250514"):
    """Get a configured Claude LLM instance.

    Args:
        temperature: Sampling temperature (0.0 = deterministic).
        model: Anthropic model identifier.

    Returns:
        A ChatAnthropic instance ready to use.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Set it in your .env file or environment variables."
        )

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=api_key,
        max_tokens=4096,
    )


def call_llm(prompt: str, response_model=None, temperature: float = 0.0):
    """Call Claude with a prompt and optionally parse into a Pydantic model.

    Args:
        prompt: The prompt to send to Claude.
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
