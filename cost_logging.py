import logging

logger = logging.getLogger("takanot_rag")

OPENAI_LLM_PRICING = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.0006, "completion": 0.0018},
    "gpt-3.5-turbo-0125": {"prompt": 0.0005, "completion": 0.0015},
    # Add more models as needed
}


def log_llm_cost(response, model_name):
    try:
        usage = None
        # Try both possible locations for usage info
        if hasattr(response, "response_metadata") and response.response_metadata:
            usage = response.response_metadata.get("token_usage")
        if not usage and hasattr(response, "additional_kwargs"):
            usage = response.additional_kwargs.get("usage")
        if usage:
            # Robust extraction for dict, attribute, or other types
            def extract(key, default=0):
                try:
                    if isinstance(usage, dict):
                        return usage.get(key, default)
                    return getattr(usage, key, default)
                except Exception:
                    try:
                        return usage[key]
                    except Exception:
                        return default

            prompt = extract("prompt_tokens", 0)
            completion = extract("completion_tokens", 0)
            total = extract("total_tokens", 0)
            price = OPENAI_LLM_PRICING.get(model_name, {"prompt": 0, "completion": 0})
            cost = (prompt * price["prompt"] + completion * price["completion"]) / 1000
            logger.info(
                f"OpenAI LLM | model={model_name} | prompt_tokens={prompt} | completion_tokens={completion} | total_tokens={total} | cost_usd={cost:.8f}"
            )
        else:
            logger.warning(
                f"log_llm_cost: No usage info found in response for model={model_name}"
            )
    except Exception as e:
        logger.error(
            f"log_llm_cost: Exception occurred: {e} | usage type: {type(usage)} | usage repr: {repr(usage)}"
        )
