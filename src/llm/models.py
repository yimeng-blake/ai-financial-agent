"""
LLM client configuration — MiniMax as the primary model.

Uses manual JSON parsing instead of with_structured_output() because
MiniMax does not reliably support LangChain's tool-calling-based
structured output protocol.
"""

import json
import logging
import os
import re

from dotenv import load_dotenv
from langchain_community.chat_models import MiniMaxChat

load_dotenv()

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def get_llm(temperature: float = 0.0, model: str = "MiniMax-Text-01"):
    """Get a configured MiniMax LLM instance.

    Args:
        temperature: Sampling temperature (0.0 = deterministic).
        model: MiniMax model identifier.

    Returns:
        A MiniMaxChat instance ready to use.
    """
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise ValueError(
            "MINIMAX_API_KEY not found. "
            "Set it in your .env file or environment variables."
        )

    return MiniMaxChat(
        model=model,
        temperature=temperature,
        minimax_api_key=api_key,
        max_tokens=4096,
    )


def _sanitize_json_string(text: str) -> str:
    """Sanitize a JSON string by escaping unescaped control characters.

    LLMs often produce JSON with literal newlines/tabs inside string values
    which is invalid per the JSON spec.  This replaces them so json.loads()
    can handle the output.
    """
    # Replace literal control characters that are not already escaped
    text = text.replace("\r\n", "\\n").replace("\r", "\\n")
    # Replace literal newlines inside strings — but NOT the structural ones
    # Strategy: replace all \n with \\n, then restore structural ones
    # by re-parsing. Simpler: just use strict=False in json.loads.
    return text


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    # Try to find JSON in ```json ... ``` or ``` ... ``` blocks
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find a top-level JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0).strip()

    return text.strip()


def _build_schema_prompt(response_model) -> str:
    """Build a JSON schema description from a Pydantic model for the prompt."""
    schema = response_model.model_json_schema()
    properties = schema.get("properties", {})

    fields_desc = []
    for name, prop in properties.items():
        desc = prop.get("description", "")
        ptype = prop.get("type", "string")
        fields_desc.append(f'  "{name}": ({ptype}) {desc}')

    return (
        "你必须以纯 JSON 格式回复，不要包含任何其他文字。\n"
        "JSON 对象的字段如下：\n"
        + "\n".join(fields_desc)
    )


def call_llm(prompt: str, response_model=None, temperature: float = 0.0):
    """Call MiniMax with a prompt and optionally parse into a Pydantic model.

    For structured output, appends JSON schema instructions to the prompt
    and parses the response manually, since MiniMax does not support
    LangChain's with_structured_output() reliably.

    Args:
        prompt: The prompt to send.
        response_model: Optional Pydantic model class for structured output.
        temperature: Sampling temperature.

    Returns:
        If response_model is provided, returns a parsed Pydantic object.
        Otherwise, returns the raw string response.
    """
    llm = get_llm(temperature=temperature)

    if not response_model:
        response = llm.invoke(prompt)
        return response.content

    # --- Structured output via JSON prompting + manual parsing ---
    schema_instruction = _build_schema_prompt(response_model)
    full_prompt = f"{prompt}\n\n{schema_instruction}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke(full_prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            json_str = _extract_json(raw)
            # strict=False allows control chars (newlines, tabs) inside
            # JSON string values — very common in LLM output.
            data = json.loads(json_str, strict=False)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "Structured output parse failed (attempt %d/%d) for %s: %s",
                attempt, MAX_RETRIES, response_model.__name__, exc,
            )
            if attempt < MAX_RETRIES:
                # Retry with a stronger nudge
                full_prompt = (
                    f"{prompt}\n\n"
                    f"{schema_instruction}\n\n"
                    "重要：只输出合法的 JSON，不要输出任何解释文字。"
                    "字符串值中的换行符必须用 \\n 转义。"
                )
            else:
                logger.error(
                    "All %d attempts failed for %s. Raw response: %s",
                    MAX_RETRIES, response_model.__name__, raw[:500],
                )
                raise ValueError(
                    f"Failed to parse {response_model.__name__} from LLM "
                    f"after {MAX_RETRIES} attempts: {exc}"
                ) from exc
