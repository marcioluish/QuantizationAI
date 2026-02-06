"""
Meeting Minutes Generator - Gradio Application
Main UI application for transcribing audio and generating meeting minutes.
"""

import os
from typing import Dict, List, Optional, Tuple

import gradio as gr

from model_configs import (
    get_model_choices,
    get_model_info,
    get_model_display_name,
    AVAILABLE_MODELS
)
from quantization import (
    get_bits_options,
    get_double_quant_options,
    get_quant_type_options,
    get_default_config
)
from transcription import transcribe_audio, clear_transcription_cache
from minute_generation import generate_minutes
from stats import (
    StatisticsCollector,
    create_performance_charts,
    create_memory_charts
)
from analysis import analyze_minutes
from utils import (
    validate_audio_file,
    validate_model_selection,
    clear_gpu_memory,
    log_gpu_memory,
    logger
)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class AppState:
    """Global application state."""
    def __init__(self):
        self.is_processing = False
        self.cancel_requested = False
        self.hf_token: Optional[str] = None
        self.openai_key: Optional[str] = None
        self.stats_collector: Optional[StatisticsCollector] = None
        self.results: Dict[str, str] = {}   # model_id -> generated minutes
        self.errors: Dict[str, str] = {}    # model_id -> error message


app_state = AppState()

# Registries populated by create_app() and consumed by event handlers
MODEL_COMPONENTS: Dict[str, Dict[str, gr.Component]] = {}
MODEL_TABS: Dict[str, gr.Tab] = {}
MODEL_ORDER: List[str] = []            # all model IDs in display order
OUTPUT_COMPONENTS: List[gr.Component] = []
OUTPUT_INDEX: Dict[int, int] = {}      # id(component) -> index


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def progress_html(percent: int, status: str) -> str:
    """Return HTML for a simple progress bar with status text."""
    percent = max(0, min(100, int(percent)))
    return (
        f"<div style='margin-bottom:6px;'>"
        f"<progress value='{percent}' max='100' style='width:100%;height:22px;'></progress>"
        f"</div><div style='font-size:0.9em;'>{status}</div>"
    )


def _blank_updates():
    """Create a list of gr.update() for every output component."""
    return [gr.update() for _ in OUTPUT_COMPONENTS]


def _set(updates, component, **kwargs):
    """Set a component update in the updates list (by identity)."""
    idx = OUTPUT_INDEX.get(id(component))
    if idx is not None:
        updates[idx] = gr.update(**kwargs)


# ---------------------------------------------------------------------------
# Token initialisation
# ---------------------------------------------------------------------------

def initialize_tokens():
    """Load API tokens from Colab secrets or environment variables."""
    try:
        from google.colab import userdata
        app_state.hf_token = userdata.get('HF_TOKEN')
        app_state.openai_key = userdata.get('OPENAI_API_KEY')
        logger.info("Loaded tokens from Colab secrets")
    except Exception:
        app_state.hf_token = os.environ.get('HF_TOKEN')
        app_state.openai_key = os.environ.get('OPENAI_API_KEY')
        logger.info("Loaded tokens from environment variables")

    if app_state.hf_token:
        try:
            from huggingface_hub import login
            login(app_state.hf_token, add_to_git_credential=False)
            logger.info("Logged in to Hugging Face")
        except Exception as e:
            logger.warning(f"Failed to login to Hugging Face: {e}")


# ---------------------------------------------------------------------------
# Quantization dropdown helpers
# ---------------------------------------------------------------------------

def update_double_quant_options(bits: str):
    options, enabled, reason = get_double_quant_options(bits)
    if enabled:
        return (
            gr.Dropdown(choices=options, value=options[0], interactive=True),
            gr.Markdown(visible=False),
        )
    return (
        gr.Dropdown(choices=options, value=options[0], interactive=False),
        gr.Markdown(value=f"*{reason}*", visible=True),
    )


def update_quant_type_options(bits: str):
    options, enabled, reason = get_quant_type_options(bits)
    if enabled:
        return (
            gr.Dropdown(choices=options, value=options[0], interactive=True),
            gr.Markdown(visible=False),
        )
    return (
        gr.Dropdown(choices=options, value=options[0], interactive=False),
        gr.Markdown(value=f"*{reason}*", visible=True),
    )


# ---------------------------------------------------------------------------
# Model-info markdown (shown in each result tab)
# ---------------------------------------------------------------------------

def get_model_info_html(model_id: str) -> str:
    info = get_model_info(model_id)
    if not info:
        return ""
    return (
        f"### Model Information\n\n"
        f"| Property | Value |\n"
        f"|----------|-------|\n"
        f"| **Organization** | {info.organization} |\n"
        f"| **Parameters** | {info.parameters} |\n"
        f"| **Size (FP16)** | {info.size_fp16} |\n"
        f"| **Size (4-bit)** | {info.size_4bit} |\n"
        f"| **Fine-tuned** | {'Yes - ' + info.fine_tune_method if info.is_fine_tuned else 'No'} |\n"
        f"\n{info.description}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(audio_file, selected_models):
    ok, err = validate_audio_file(audio_file)
    if not ok:
        return False, err
    ok, err = validate_model_selection(selected_models)
    if not ok:
        return False, err
    return True, ""


# ---------------------------------------------------------------------------
# Main generation handler  (generator – yields update lists)
# ---------------------------------------------------------------------------

def on_generate_click(
    audio_file: Optional[str],
    selected_models: List[str],
    bits: str,
    double_quant: str,
    quant_type: str,
    progress=gr.Progress(),
):
    # ---- validate --------------------------------------------------------
    ok, err = validate_inputs(audio_file, selected_models)
    if not ok:
        u = _blank_updates()
        _set(u, error_display, value=f"**Error:** {err}", visible=True)
        _set(u, generate_btn, value="Generate Minutes", variant="primary")
        yield u
        return

    # ---- initialise state ------------------------------------------------
    app_state.is_processing = True
    app_state.cancel_requested = False
    app_state.stats_collector = StatisticsCollector()
    app_state.results = {}
    app_state.errors = {}

    try:
        # -- hide all model tabs & stats, show progress --------------------
        u = _blank_updates()
        _set(u, error_display, visible=False)
        _set(u, generate_btn, value="Cancel", variant="stop")
        _set(u, transcription_progress, value="Starting transcription...", visible=True)
        _set(u, transcription_accordion, visible=False)
        _set(u, analysis_output, visible=False)
        _set(u, stats_tab_col, visible=False)
        _set(u, stats_notice, visible=False)
        _set(u, stats_group, visible=False)

        # hide every model tab
        for mid in MODEL_ORDER:
            _set(u, MODEL_TABS[mid], visible=False)
            _set(u, MODEL_COMPONENTS[mid]["status"], value="")
            _set(u, MODEL_COMPONENTS[mid]["progress"], value="")
            _set(u, MODEL_COMPONENTS[mid]["minutes"], value="")
        yield u

        # ---- cancellation check ------------------------------------------
        if app_state.cancel_requested:
            yield _handle_cancellation()
            return

        # ==================================================================
        # STEP 1  –  Transcription
        # ==================================================================
        log_gpu_memory("before transcription step")
        logger.info("Starting transcription")

        def transcription_cb(pct, msg):
            progress(pct * 0.2, desc=msg)  # 0-20 %

        transcription, trans_err = transcribe_audio(
            audio_file, progress_callback=transcription_cb
        )

        if trans_err:
            u = _blank_updates()
            _set(u, error_display, value=f"**Transcription Error:** {trans_err}", visible=True)
            _set(u, generate_btn, value="Generate Minutes", variant="primary")
            _set(u, transcription_progress, visible=False)
            yield u
            return

        # show collapsed transcription
        u = _blank_updates()
        _set(u, transcription_progress, value="Transcription complete!", visible=True)
        _set(u, transcription_accordion, label="Transcription (click to expand)", open=False, visible=True)
        _set(u, transcription_display, value=transcription)
        yield u

        if app_state.cancel_requested:
            yield _handle_cancellation()
            return

        # clean GPU after Whisper
        clear_gpu_memory("after Whisper")

        # ==================================================================
        # STEP 2  –  Sequential minute generation
        # ==================================================================
        quant_cfg = {"bits": bits, "double_quant": double_quant, "quant_type": quant_type}
        n_models = len(selected_models)
        logger.info(f"Generating minutes for {n_models} model(s) sequentially")

        for i, model_id in enumerate(selected_models):
            if app_state.cancel_requested:
                yield _handle_cancellation()
                return

            display_name = get_model_display_name(model_id)

            # -- reveal this model's tab with "loading" state ---------------
            u = _blank_updates()
            _set(u, MODEL_TABS[model_id], visible=True)
            _set(u, MODEL_COMPONENTS[model_id]["status"], value=f"Loading {display_name}...")
            _set(u, MODEL_COMPONENTS[model_id]["progress"], value=progress_html(5, "Loading model..."))
            _set(u, MODEL_COMPONENTS[model_id]["minutes"], value="")
            yield u

            # progress callback
            def _make_cb(idx, name):
                def cb(pct, msg):
                    base = 0.2 + (0.7 / n_models) * idx
                    portion = 0.7 / n_models
                    progress(base + pct * portion, desc=f"{name}: {msg}")
                return cb

            model_cb = _make_cb(i, display_name)
            logger.info(f"Processing model {i+1}/{n_models}: {model_id}")

            minutes, error, stats = generate_minutes(
                model_id=model_id,
                transcription=transcription,
                quant_config=quant_cfg,
                stats_collector=app_state.stats_collector,
                hf_token=app_state.hf_token,
                progress_callback=model_cb,
            )

            # GPU cleanup happens inside generate_minutes already,
            # but do an extra pass to be safe
            clear_gpu_memory(f"after model {i+1}/{n_models} - {model_id}")

            u = _blank_updates()
            if error:
                app_state.errors[model_id] = error
                logger.error(f"Error for {model_id}: {error}")
                _set(u, MODEL_COMPONENTS[model_id]["status"], value="Generation failed.")
                _set(u, MODEL_COMPONENTS[model_id]["progress"], value=progress_html(100, "Error"))
                _set(u, MODEL_COMPONENTS[model_id]["minutes"], value=f"**Error:** {error}")
            else:
                app_state.results[model_id] = minutes
                logger.info(f"Success for {model_id}")
                _set(u, MODEL_COMPONENTS[model_id]["status"], value="Generation complete!")
                _set(u, MODEL_COMPONENTS[model_id]["progress"], value=progress_html(100, "Complete"))
                _set(u, MODEL_COMPONENTS[model_id]["minutes"], value=minutes)
            yield u

        # ==================================================================
        # STEP 3  –  Statistics tab  (only if >= 1 model succeeded)
        # ==================================================================
        successful = app_state.stats_collector.get_successful_stats()
        u = _blank_updates()
        if successful:
            _set(u, stats_tab_col, visible=True)
            _set(u, stats_group, visible=True)
            _set(u, stats_notice, visible=False)
            _set(u, performance_chart, value=create_performance_charts(successful))
            _set(u, memory_chart, value=create_memory_charts(successful))
        else:
            _set(u, stats_tab_col, visible=True)
            _set(u, stats_notice, value="No statistics available — all models failed.", visible=True)
            _set(u, stats_group, visible=False)
        yield u

        # ==================================================================
        # STEP 4  –  GPT-4o-mini analysis  (only if >= 2 succeeded)
        # ==================================================================
        progress(0.95, desc="Analyzing results...")
        analysis_text = ""
        if len(app_state.results) >= 2 and app_state.openai_key:
            names = {mid: get_model_display_name(mid) for mid in app_state.results}
            analysis_text = analyze_minutes(app_state.results, names, app_state.openai_key)

        progress(1.0, desc="Complete!")

        u = _blank_updates()
        _set(u, generate_btn, value="Generate Minutes", variant="primary")
        _set(u, transcription_progress, visible=False)
        if analysis_text:
            _set(u, analysis_output, value=analysis_text, visible=True)
        else:
            _set(u, analysis_output, visible=False)
        yield u

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        u = _blank_updates()
        _set(u, error_display, value=f"**Unexpected Error:** {str(e)}", visible=True)
        _set(u, generate_btn, value="Generate Minutes", variant="primary")
        yield u
    finally:
        app_state.is_processing = False


def _handle_cancellation():
    logger.info("Processing cancelled by user")
    clear_gpu_memory()
    clear_transcription_cache()
    u = _blank_updates()
    _set(u, error_display, value="**Processing cancelled.** All progress has been lost.", visible=True)
    _set(u, generate_btn, value="Generate Minutes", variant="primary")
    _set(u, transcription_progress, visible=False)
    return u


# ---------------------------------------------------------------------------
# Token init on import
# ---------------------------------------------------------------------------
initialize_tokens()


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------

def create_app():
    """Build and return the Gradio Blocks application."""

    # globals written here, read by the generator above
    global error_display, generate_btn, transcription_progress
    global transcription_accordion, transcription_display, analysis_output
    global performance_chart, memory_chart, stats_group, stats_notice, stats_tab_col

    defaults = get_default_config()
    model_choices = get_model_choices()

    css = """
    .gradio-container { max-width: 1200px !important; }
    .model-warning   { color: #ffa500; font-size: 0.9em; }
    .disabled-reason  { color: #888; font-style: italic; font-size: 0.85em; margin-top: 4px; }
    .time-warning     { color: #ccc; font-size: 0.85em; margin-top: 6px; }
    """

    with gr.Blocks(
        title="Meeting Minutes Generator",
        theme=gr.themes.Soft(primary_hue="blue").set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
        ),
        css=css,
    ) as app:

        gr.Markdown(
            "# Meeting Minutes Generator\n\n"
            "Upload an audio file and generate professional meeting minutes "
            "using various AI models.  Compare outputs and get AI-powered analysis."
        )

        with gr.Tabs() as main_tabs:

            # =============================================================
            # TAB: Configuration
            # =============================================================
            with gr.Tab("Configuration", id="config_tab"):

                error_display = gr.Markdown(visible=False)

                # -- file upload ------------------------------------------
                with gr.Row():
                    audio_input = gr.Audio(
                        label="Upload Audio File (MP3, max 50MB)",
                        type="filepath",
                        sources=["upload"],
                    )

                # -- model selection + quantization -----------------------
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Model Selection")
                        model_selector = gr.Dropdown(
                            label="Select Models",
                            choices=model_choices,
                            multiselect=True,
                            info="Select the models you wish to compare when generating minutes",
                        )
                        gr.Markdown(
                            "*⚠️ Some models require access approval. "
                            "Visit the model page on Hugging Face to request access before using.*",
                            elem_classes=["model-warning"],
                        )
                        selected_models_display = gr.Markdown("")
                        time_warning_display = gr.Markdown("", elem_classes=["time-warning"])

                    with gr.Column(scale=1):
                        gr.Markdown("### Quantization Settings")
                        bits_dropdown = gr.Dropdown(
                            label="Bits",
                            choices=get_bits_options(),
                            value=defaults["bits"],
                            info="Number of bits for quantization",
                        )
                        double_quant_dropdown = gr.Dropdown(
                            label="Double Quantization",
                            choices=get_double_quant_options(defaults["bits"])[0],
                            value=defaults["double_quant"],
                            info="Apply double quantization for more compression",
                        )
                        double_quant_reason = gr.Markdown(visible=False, elem_classes=["disabled-reason"])
                        quant_type_dropdown = gr.Dropdown(
                            label="Quantization Type",
                            choices=get_quant_type_options(defaults["bits"])[0],
                            value=defaults["quant_type"],
                            info="Type of 4-bit quantization",
                        )
                        quant_type_reason = gr.Markdown(visible=False, elem_classes=["disabled-reason"])

                # -- generate button --------------------------------------
                with gr.Row():
                    generate_btn = gr.Button("Generate Minutes", variant="primary", scale=2)

                # -- transcription progress / result ----------------------
                transcription_progress = gr.Markdown(visible=False)
                transcription_accordion = gr.Accordion(
                    label="Transcription (click to expand)", open=False, visible=False
                )
                with transcription_accordion:
                    transcription_display = gr.Markdown()

                # -- GPT-4o-mini analysis ---------------------------------
                gr.Markdown("### AI Analysis")
                analysis_output = gr.Markdown(visible=False)

            # =============================================================
            # TABS: one per model (all hidden at start)
            # =============================================================
            MODEL_COMPONENTS.clear()
            MODEL_TABS.clear()
            MODEL_ORDER.clear()

            for model_id in model_choices:
                display_name = get_model_display_name(model_id)
                MODEL_ORDER.append(model_id)

                tab = gr.Tab(display_name, visible=False)
                MODEL_TABS[model_id] = tab

                with tab:
                    gr.Markdown(f"## {display_name}")
                    m_info = gr.Markdown(get_model_info_html(model_id))
                    m_status = gr.Markdown("")
                    m_progress = gr.HTML("")
                    gr.Markdown("### Generated Minutes")
                    m_minutes = gr.Markdown("")

                    MODEL_COMPONENTS[model_id] = {
                        "info": m_info,
                        "status": m_status,
                        "progress": m_progress,
                        "minutes": m_minutes,
                    }

            # =============================================================
            # TAB: Statistics (hidden until results are ready)
            # =============================================================
            stats_tab_col = gr.Tab("Statistics", visible=False)
            with stats_tab_col:
                stats_notice = gr.Markdown(visible=False)
                stats_group = gr.Group(visible=False)
                with stats_group:
                    with gr.Tabs():
                        with gr.Tab("Performance"):
                            performance_chart = gr.Plot(label="Performance Statistics")
                        with gr.Tab("Memory"):
                            memory_chart = gr.Plot(label="Memory Statistics")

        # =================================================================
        # Event wiring
        # =================================================================

        # -- model-selection display + time warning -----------------------
        def _on_model_change(models):
            if not models:
                return "", ""
            names = ", ".join(get_model_display_name(m) for m in models)
            n = len(models)
            warning = (
                f"*Processing {n} model(s) sequentially may take approximately "
                f"{n * 3}–{n * 8} minutes depending on model size and audio length.*"
            ) if n > 1 else ""
            return f"**Selected:** {names}", warning

        model_selector.change(
            fn=_on_model_change,
            inputs=[model_selector],
            outputs=[selected_models_display, time_warning_display],
        )

        # -- quantization cascading dropdowns ----------------------------
        bits_dropdown.change(
            fn=update_double_quant_options,
            inputs=[bits_dropdown],
            outputs=[double_quant_dropdown, double_quant_reason],
        ).then(
            fn=update_quant_type_options,
            inputs=[bits_dropdown],
            outputs=[quant_type_dropdown, quant_type_reason],
        )

        # -- build output list -------------------------------------------
        OUTPUT_COMPONENTS.clear()
        OUTPUT_COMPONENTS.extend([
            error_display,          # 0
            generate_btn,           # 1
            transcription_progress, # 2
            transcription_accordion,# 3
            transcription_display,  # 4
            analysis_output,        # 5
            stats_tab_col,          # 6
            stats_notice,           # 7
            stats_group,            # 8
            performance_chart,      # 9
            memory_chart,           # 10
        ])
        for mid in MODEL_ORDER:
            OUTPUT_COMPONENTS.append(MODEL_TABS[mid])          # tab visibility
            OUTPUT_COMPONENTS.append(MODEL_COMPONENTS[mid]["status"])
            OUTPUT_COMPONENTS.append(MODEL_COMPONENTS[mid]["progress"])
            OUTPUT_COMPONENTS.append(MODEL_COMPONENTS[mid]["minutes"])

        OUTPUT_INDEX.clear()
        for idx, comp in enumerate(OUTPUT_COMPONENTS):
            OUTPUT_INDEX[id(comp)] = idx

        # -- generate button -> handler ----------------------------------
        generate_btn.click(
            fn=on_generate_click,
            inputs=[
                audio_input,
                model_selector,
                bits_dropdown,
                double_quant_dropdown,
                quant_type_dropdown,
            ],
            outputs=OUTPUT_COMPONENTS,
        )

    return app


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_app():
    app = create_app()
    app.launch(share=True, debug=True)


if __name__ == "__main__":
    launch_app()
