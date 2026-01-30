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


def save_subtitle_processing_settings(
    *,
    processors: list[str] | None,
    batch_size: int,
    concurrency: int,
    target_language: str,
    split_mode: str,
    split_max_word_count_cjk: int,
    split_max_word_count_english: int,
    translate_reflect: bool,
    custom_prompt: str,
) -> None:
    allowed_processors = {"optimize", "translate", "split"}
    proc_list = [str(p).strip() for p in (processors or []) if str(p).strip()]
    proc_list = [p for p in proc_list if p in allowed_processors]
    if not proc_list:
        proc_list = ["optimize"]

    try:
        bs = int(batch_size)
    except Exception:
        bs = 30
    bs = max(1, min(200, bs))

    try:
        cc = int(concurrency)
    except Exception:
        cc = 4
    cc = max(1, min(16, cc))

    lang = (target_language or "").strip() or "zh"
    if lang not in {"zh", "en", "ja", "ko", "fr", "de", "es", "ru"}:
        lang = "zh"

    mode = (split_mode or "").strip() or "inplace_newlines"
    if mode not in {"inplace_newlines", "split_to_cues"}:
        mode = "inplace_newlines"

    try:
        mw_cjk = int(split_max_word_count_cjk)
    except Exception:
        mw_cjk = 18
    mw_cjk = max(1, min(200, mw_cjk))

    try:
        mw_en = int(split_max_word_count_english)
    except Exception:
        mw_en = 12
    mw_en = max(1, min(200, mw_en))

    update_config(
        {
            "subtitle_processors": proc_list,
            "subtitle_batch_size": bs,
            "subtitle_concurrency": cc,
            "subtitle_target_language": lang,
            "subtitle_split_mode": mode,
            "subtitle_split_max_word_count_cjk": mw_cjk,
            "subtitle_split_max_word_count_english": mw_en,
            "subtitle_translate_reflect": bool(translate_reflect),
            "subtitle_custom_prompt": str(custom_prompt or ""),
        }
    )


__all__ = ["save_subtitle_processing_settings", "save_subtitle_provider_settings"]
