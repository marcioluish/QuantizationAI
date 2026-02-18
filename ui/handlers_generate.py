"""
Processing pipeline: transcription, sequential minute generation, stats, GPT-4o analysis.
"""

import os
from typing import Optional

import gradio as gr

from ui.app_state import MAX_RESULT_SLOTS, RESULT_SLOTS, app_state
from engine.transcription import clear_transcription_cache, transcribe_audio
from engine.minute_generation import generate_minutes
from engine.stats import (
    StatisticsCollector,
    create_performance_charts,
    create_memory_charts,
)
from engine.analysis import analyze_minutes
from engine.utils import clear_gpu_memory, log_gpu_memory, logger, validate_audio_file
from ui.ui_helpers import (
    _blank_updates,
    _set,
    _btn_label,
    progress_html,
    get_model_info_html,
)

# Component refs set by app.py via register_components() before first use
_comps = {}


def register_components(components: dict):
    """Register UI component refs for the generate handler. Call once from create_app()."""
    global _comps
    _comps = components


def on_generate_click(
    audio_file: Optional[str],
    saved_configs: list,
    progress=gr.Progress(),
):
    """Main generator: validate -> transcribe -> generate per model -> stats -> analysis."""
    n_models = len(saved_configs) if saved_configs else 0
    btn_text = _btn_label(n_models)

    ok, err = validate_audio_file(audio_file)
    if not ok:
        u = _blank_updates()
        _set(u, _comps["error_display"], value=f"**Error:** {err}", visible=True)
        _set(u, _comps["generate_btn"], value=btn_text, variant="primary")
        yield u
        return

    if not saved_configs:
        u = _blank_updates()
        _set(
            u,
            _comps["error_display"],
            value=(
                "**Error:** No models configured. Use "
                "'Save Model Quantization Settings' to add at least one model."
            ),
            visible=True,
        )
        _set(u, _comps["generate_btn"], value=btn_text, variant="primary")
        yield u
        return

    app_state.is_processing = True
    app_state.cancel_requested = False
    app_state.stats_collector = StatisticsCollector()
    app_state.results = {}
    app_state.errors = {}

    try:
        u = _blank_updates()
        _set(u, _comps["error_display"], visible=False)
        _set(u, _comps["generate_btn"], value="Cancel", variant="stop")
        _set(u, _comps["transcription_progress"], value="Starting transcription...", visible=True)
        _set(u, _comps["transcription_tab"], visible=False)
        _set(u, _comps["analysis_tab"], visible=False)
        _set(u, _comps["analysis_output"], value="")
        _set(u, _comps["stats_tab_col"], visible=False)
        _set(u, _comps["stats_notice"], visible=False)
        _set(u, _comps["stats_group"], visible=False)
        for slot in RESULT_SLOTS:
            _set(u, slot["tab"], visible=False)
            _set(u, slot["info"], value="")
            _set(u, slot["status"], value="")
            _set(u, slot["progress"], value="")
            _set(u, slot["minutes"], value="")
        yield u

        if app_state.cancel_requested:
            yield _handle_cancellation(btn_text)
            return

        log_gpu_memory("before transcription step")
        logger.info("Starting transcription")

        def transcription_cb(pct, msg):
            progress(pct * 0.2, desc=msg)

        transcription, trans_err = transcribe_audio(
            audio_file, progress_callback=transcription_cb
        )

        if trans_err:
            u = _blank_updates()
            _set(u, _comps["error_display"], value=f"**Transcription Error:** {trans_err}", visible=True)
            _set(u, _comps["generate_btn"], value=btn_text, variant="primary")
            _set(u, _comps["transcription_progress"], visible=False)
            yield u
            return

        audio_name = os.path.basename(audio_file) if audio_file else "unknown"
        char_count = len(transcription)
        word_count = len(transcription.split())
        metadata_md = (
            f"**Audio File:** {audio_name}  \n"
            f"**Transcription Length:** {char_count:,} characters | "
            f"{word_count:,} words"
        )
        u = _blank_updates()
        _set(u, _comps["transcription_progress"], value="Transcription complete!", visible=True)
        _set(u, _comps["transcription_tab"], visible=True)
        _set(u, _comps["transcription_metadata"], value=metadata_md)
        _set(u, _comps["transcription_display"], value=transcription)
        yield u

        if app_state.cancel_requested:
            yield _handle_cancellation(btn_text)
            return

        clear_gpu_memory("after Whisper")
        logger.info(f"Generating minutes for {n_models} model(s) sequentially")

        for i, config in enumerate(saved_configs):
            if i >= MAX_RESULT_SLOTS:
                logger.warning(
                    f"Exceeded max result slots ({MAX_RESULT_SLOTS}), skipping remaining configs"
                )
                break
            if app_state.cancel_requested:
                yield _handle_cancellation(btn_text)
                return

            model_id = config["model_id"]
            display_name = config["display_name"]
            quant_cfg = {
                "bits": config["bits"],
                "double_quant": config["double_quant"],
                "quant_type": config["quant_type"],
            }
            unique_key = f"{model_id}_{i}"
            qt_short = config["quant_type"] if config["bits"] == "4" else "int8"
            dq_short = "DQ" if config["double_quant"] == "True" else "noDQ"
            tab_label = f"{display_name} ({config['bits']}-bit {qt_short} {dq_short})"
            stats_display = tab_label
            slot = RESULT_SLOTS[i]

            u = _blank_updates()
            _set(u, slot["tab"], visible=True, label=tab_label)
            _set(u, slot["info"], value=get_model_info_html(model_id, quant_cfg))
            _set(u, slot["status"], value=f"Loading {display_name}...")
            _set(u, slot["progress"], value=progress_html(5, "Loading model..."))
            _set(u, slot["minutes"], value="")
            yield u

            def _make_cb(idx, name):
                def cb(pct, msg):
                    base = 0.2 + (0.7 / n_models) * idx
                    portion = 0.7 / n_models
                    progress(base + pct * portion, desc=f"{name}: {msg}")
                return cb

            model_cb = _make_cb(i, display_name)
            logger.info(f"Processing model {i+1}/{n_models}: {model_id} with {quant_cfg}")

            minutes, error, _ = generate_minutes(
                model_id=model_id,
                transcription=transcription,
                quant_config=quant_cfg,
                stats_collector=app_state.stats_collector,
                hf_token=app_state.hf_token,
                progress_callback=model_cb,
                stats_key=unique_key,
                stats_display_name=stats_display,
            )

            clear_gpu_memory(f"after model {i+1}/{n_models} - {model_id}")

            u = _blank_updates()
            if error:
                app_state.errors[unique_key] = error
                logger.error(f"Error for {model_id}: {error}")
                _set(u, slot["status"], value="Generation failed.")
                _set(u, slot["progress"], value=progress_html(100, "Error"))
                _set(u, slot["minutes"], value=f"**Error:** {error}")
            else:
                app_state.results[unique_key] = minutes
                logger.info(f"Success for {model_id}")
                _set(u, slot["status"], value="Generation complete!")
                _set(u, slot["progress"], value=progress_html(100, "Complete"))
                _set(u, slot["minutes"], value=minutes)
            yield u

        successful = app_state.stats_collector.get_successful_stats()
        u = _blank_updates()
        if successful:
            _set(u, _comps["stats_tab_col"], visible=True)
            _set(u, _comps["stats_group"], visible=True)
            _set(u, _comps["stats_notice"], visible=False)
            _set(u, _comps["performance_chart"], value=create_performance_charts(successful))
            _set(u, _comps["memory_chart"], value=create_memory_charts(successful))
        else:
            _set(u, _comps["stats_tab_col"], visible=True)
            _set(u, _comps["stats_notice"], value="No statistics available — all models failed.", visible=True)
            _set(u, _comps["stats_group"], visible=False)
        yield u

        progress(0.95, desc="Analyzing results...")
        analysis_text = ""
        if len(app_state.results) >= 2 and app_state.openai_key:
            names = {}
            for j, cfg in enumerate(saved_configs):
                ukey = f"{cfg['model_id']}_{j}"
                if ukey in app_state.results:
                    qt_s = cfg["quant_type"] if cfg["bits"] == "4" else "int8"
                    dq_s = "DQ" if cfg["double_quant"] == "True" else "noDQ"
                    names[ukey] = f"{cfg['display_name']} ({cfg['bits']}-bit {qt_s} {dq_s})"
            analysis_text = analyze_minutes(app_state.results, names, app_state.openai_key)

        progress(1.0, desc="Complete!")
        u = _blank_updates()
        _set(u, _comps["generate_btn"], value=btn_text, variant="primary")
        _set(u, _comps["transcription_progress"], visible=False)
        if analysis_text:
            _set(u, _comps["analysis_tab"], visible=True)
            _set(u, _comps["analysis_output"], value=analysis_text)
        else:
            _set(u, _comps["analysis_tab"], visible=False)
        yield u

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        u = _blank_updates()
        _set(u, _comps["error_display"], value=f"**Unexpected Error:** {str(e)}", visible=True)
        _set(u, _comps["generate_btn"], value=btn_text, variant="primary")
        yield u
    finally:
        app_state.is_processing = False


def _handle_cancellation(btn_text: str = "Generate Minute"):
    """Build update list for cancellation: clear state, show message, hide tabs."""
    logger.info("Processing cancelled by user")
    clear_gpu_memory()
    clear_transcription_cache()
    u = _blank_updates()
    _set(u, _comps["error_display"], value="**Processing cancelled.** All progress has been lost.", visible=True)
    _set(u, _comps["generate_btn"], value=btn_text, variant="primary")
    _set(u, _comps["transcription_progress"], visible=False)
    _set(u, _comps["transcription_tab"], visible=False)
    _set(u, _comps["analysis_tab"], visible=False)
    for slot in RESULT_SLOTS:
        _set(u, slot["tab"], visible=False)
    return u
