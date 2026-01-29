from __future__ import annotations

from auto_asr.config import update_config


def save_subtitle_provider_settings(
    *,
    provider: str,
    openai_api_key: str,
    openai_base_url: str,
    llm_model: str,
    llm_temperature: float,
    split_strategy: str,
) -> None:
    provider = (provider or "").strip() or "openai"
    split_strategy = (split_strategy or "").strip() or "semantic"
    if split_strategy not in {"semantic", "sentence"}:
        split_strategy = "semantic"

    try:
        temperature = float(llm_temperature)
    except Exception:
        temperature = 0.2
    temperature = max(0.0, min(2.0, temperature))

    update_config(
        {
            "subtitle_provider": provider,
            "subtitle_openai_api_key": (openai_api_key or "").strip(),
            "subtitle_openai_base_url": (openai_base_url or "").strip(),
            "subtitle_llm_model": (llm_model or "").strip() or "gpt-4o-mini",
            "subtitle_llm_temperature": temperature,
            "subtitle_split_strategy": split_strategy,
        }
    )


__all__ = ["save_subtitle_provider_settings"]
