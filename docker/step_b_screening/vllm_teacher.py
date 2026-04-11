"""vLLM-based Teacher VLM inference for helmet chinstrap screening.

Uses Qwen3.5-27B (natively multimodal) via vLLM offline LLM class.
Structured output is enforced via StructuredOutputsParams (JSON schema).
"""

import json
from pathlib import Path


TEACHER_SYSTEM_PROMPT = (
    "You are a safety inspection assistant. "
    "Look at the image and determine the helmet chinstrap status."
)

TEACHER_USER_PROMPT = (
    "Determine if the helmet chinstrap in this image is tight or loose. "
    "Respond ONLY in the following JSON format:\n"
    '{"label": "tight" or "loose", "rationale": "<one sentence explaining visible evidence>"}'
)

LABEL_SCHEMA = {
    "type": "object",
    "required": ["label", "rationale"],
    "properties": {
        "label": {"type": "string", "enum": ["tight", "loose"]},
        "rationale": {"type": "string"},
    },
    "additionalProperties": False,
}

LABEL_ONLY_SCHEMA = {
    "type": "object",
    "required": ["label"],
    "properties": {
        "label": {"type": "string", "enum": ["tight", "loose"]},
    },
    "additionalProperties": False,
}


def _build_messages(image_path: Path, rationale: bool = True) -> list[dict]:
    """Build OpenAI-style chat messages for a single image."""
    user_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"file://{image_path}"},
        },
        {
            "type": "text",
            "text": TEACHER_USER_PROMPT if rationale else (
                "Determine if the helmet chinstrap in this image is tight or loose. "
                'Respond ONLY in JSON: {"label": "tight" or "loose"}'
            ),
        },
    ]

    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def run_teacher_inference(
    image_paths: list[Path],
    image_ids: list[str],
    model_id: str = "Qwen/Qwen3.5-27B",
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    enable_thinking: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 256,
    rationale: bool = True,
    dtype: str = "auto",
) -> dict[str, dict]:
    """Run Teacher VLM on a batch of images using vLLM offline inference.

    Args:
        image_paths: List of paths to images.
        image_ids: Corresponding image IDs (same order as image_paths).
        model_id: HuggingFace model ID.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        max_model_len: Max context length.
        enable_thinking: Whether to enable Qwen3.5 thinking mode.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_tokens: Max output tokens.
        rationale: Whether to request rationale in output.
        dtype: Model dtype — "auto", "float16", "bfloat16", or "float8" for
            FP8 quantization (needed when VRAM < model size at full precision).

    Returns:
        Dict mapping image_id -> {"label": "tight"|"loose", "rationale": "..."}.
    """
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    # Build LLM kwargs; only pass quantization when explicitly requested
    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
    }
    if dtype != "auto":
        llm_kwargs["dtype"] = dtype

    # Load model once
    llm = LLM(**llm_kwargs)

    # Configure structured output
    schema = LABEL_SCHEMA if rationale else LABEL_ONLY_SCHEMA
    structured_outputs = StructuredOutputsParams(json=schema)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        structured_outputs=structured_outputs,
    )

    # Build message batches
    conversations = []
    for image_path in image_paths:
        messages = _build_messages(image_path, rationale=rationale)
        conversations.append(messages)

    # Run batch inference
    chat_template_kwargs = {}
    if not enable_thinking:
        chat_template_kwargs["enable_thinking"] = False

    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
        chat_template_kwargs=chat_template_kwargs if chat_template_kwargs else None,
    )

    # Parse results
    results = {}
    for image_id, output in zip(image_ids, outputs):
        text = output.outputs[0].text.strip()
        try:
            parsed = json.loads(text)
            results[image_id] = {
                "label": parsed["label"],
                "rationale": parsed.get("rationale", ""),
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: failed to parse output for {image_id}: {e}")
            print(f"  Raw output: {text!r}")
            # Skip unparseable results — they won't appear in teacher_responses
            # and will be excluded from screening (same as missing response)
            continue

    return results
