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
    get_default_config,
    TOOLTIP_BITS,
    TOOLTIP_DOUBLE_QUANT,
    TOOLTIP_QUANT_TYPE,
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
    clear_gpu_memory,
    log_gpu_memory,
    logger
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RESULT_SLOTS = 10   # pre-created hidden result tabs


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
        self.results: Dict[str, str] = {}   # unique_key -> generated minutes
        self.errors: Dict[str, str] = {}    # unique_key -> error message


app_state = AppState()

# Registries populated by create_app() and consumed by event handlers
RESULT_SLOTS: List[Dict[str, gr.Component]] = []
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


def _btn_label(n_models: int) -> str:
    """Return the correct Generate button label based on model count."""
    return "Generate Minute" if n_models <= 1 else "Generate Minutes"


def _tooltip_html(label: str, tooltip: str) -> str:
    """Render a label with an info tooltip icon (hover).

    Pure-CSS approach: hovering the icon reveals the explanation
    below as a normal block element (no absolute/fixed positioning),
    so it is never clipped by Gradio's overflow:hidden containers.
    """
    return (
        f'<div class="tt-container">'
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<span style="font-weight:600;font-size:0.95em;">{label}</span>'
        f'<span class="tt-icon">i</span>'
        f'</div>'
        f'<div class="tt-content">{tooltip}</div>'
        f'</div>'
    )


def _render_saved_models(saved_configs: list) -> str:
    """Render HTML for the selected models panel."""
    if not saved_configs:
        return (
            '<div style="color:#888;font-style:italic;padding:12px 0;">'
            'No models saved yet. Select a model and quantization '
            'settings, then click &ldquo;Save Model Quantization '
            'Settings&rdquo;.</div>'
        )

    html = ""
    for cfg in saved_configs:
        qt_display = cfg["quant_type"] if cfg["bits"] == "4" else "int8"
        html += (
            '<div style="display:flex;justify-content:space-between;'
            'align-items:flex-start;padding:10px 14px;margin-bottom:8px;'
            'border-radius:6px;background:rgba(255,255,255,0.06);'
            'border:1px solid rgba(255,255,255,0.1);">'
            '<div>'
            f'<div style="font-weight:600;font-size:0.95em;">'
            f'{cfg["display_name"]}</div>'
            f'<div style="font-size:0.82em;color:#aaa;margin-top:3px;">'
            f'Bits={cfg["bits"]} &nbsp;|&nbsp; '
            f'Double Quantization={cfg["double_quant"]} &nbsp;|&nbsp; '
            f'Quantization Type={qt_display}'
            f'</div>'
            '</div>'
            '</div>'
        )
    return html


def _remove_choices(saved_configs: list) -> List[str]:
    """Build the choices list for the remove dropdown."""
    choices = []
    for i, cfg in enumerate(saved_configs):
        bits_label = f'{cfg["bits"]}-bit'
        dq_label = f'DQ={"Yes" if cfg["double_quant"] == "True" else "No"}'
        qt_label = cfg["quant_type"] if cfg["bits"] == "4" else "int8"
        choices.append(
            f'{i + 1}. {cfg["display_name"]} ({bits_label}, {dq_label}, {qt_label})'
        )
    return choices


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

def get_model_info_html(
    model_id: str, quant_cfg: Optional[Dict[str, str]] = None
) -> str:
    info = get_model_info(model_id)
    if not info:
        return ""
    md = (
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
    if quant_cfg:
        qt_display = quant_cfg["quant_type"] if quant_cfg["bits"] == "4" else "int8"
        md += (
            f"\n\n### Quantization Configuration\n\n"
            f"| Setting | Value |\n"
            f"|---------|-------|\n"
            f"| **Bits** | {quant_cfg['bits']} |\n"
            f"| **Double Quantization** | {quant_cfg['double_quant']} |\n"
            f"| **Quantization Type** | {qt_display} |\n"
        )
    return md


# ---------------------------------------------------------------------------
# Main generation handler  (generator – yields update lists)
# ---------------------------------------------------------------------------

def on_generate_click(
    audio_file: Optional[str],
    saved_configs: list,
    progress=gr.Progress(),
):
    n_models = len(saved_configs) if saved_configs else 0
    btn_text = _btn_label(n_models)

    # ---- validate --------------------------------------------------------
    ok, err = validate_audio_file(audio_file)
    if not ok:
        u = _blank_updates()
        _set(u, error_display, value=f"**Error:** {err}", visible=True)
        _set(u, generate_btn, value=btn_text, variant="primary")
        yield u
        return

    if not saved_configs:
        u = _blank_updates()
        _set(
            u, error_display,
            value=(
                "**Error:** No models configured. Use "
                "'Save Model Quantization Settings' to add at least one model."
            ),
            visible=True,
        )
        _set(u, generate_btn, value=btn_text, variant="primary")
        yield u
        return

    # ---- initialise state ------------------------------------------------
    app_state.is_processing = True
    app_state.cancel_requested = False
    app_state.stats_collector = StatisticsCollector()
    app_state.results = {}
    app_state.errors = {}

    try:
        # -- hide all result slots & stats, show progress ------------------
        u = _blank_updates()
        _set(u, error_display, visible=False)
        _set(u, generate_btn, value="Cancel", variant="stop")
        _set(u, transcription_progress,
             value="Starting transcription...", visible=True)
        _set(u, transcription_tab, visible=False)
        _set(u, analysis_tab, visible=False)
        _set(u, analysis_output, value="")
        _set(u, stats_tab_col, visible=False)
        _set(u, stats_notice, visible=False)
        _set(u, stats_group, visible=False)

        for slot in RESULT_SLOTS:
            _set(u, slot["tab"], visible=False)
            _set(u, slot["info"], value="")
            _set(u, slot["status"], value="")
            _set(u, slot["progress"], value="")
            _set(u, slot["minutes"], value="")
        yield u

        # ---- cancellation check ------------------------------------------
        if app_state.cancel_requested:
            yield _handle_cancellation(btn_text)
            return

        # ==================================================================
        # STEP 1  –  Transcription
        # ==================================================================
        log_gpu_memory("before transcription step")
        logger.info("Starting transcription")

        def transcription_cb(pct, msg):
            progress(pct * 0.2, desc=msg)

        transcription, trans_err = transcribe_audio(
            audio_file, progress_callback=transcription_cb
        )

        if trans_err:
            u = _blank_updates()
            _set(u, error_display,
                 value=f"**Transcription Error:** {trans_err}", visible=True)
            _set(u, generate_btn, value=btn_text, variant="primary")
            _set(u, transcription_progress, visible=False)
            yield u
            return

        # show transcription in its own tab
        audio_name = os.path.basename(audio_file) if audio_file else "unknown"
        char_count = len(transcription)
        word_count = len(transcription.split())
        metadata_md = (
            f"**Audio File:** {audio_name}  \n"
            f"**Transcription Length:** {char_count:,} characters | "
            f"{word_count:,} words"
        )

        u = _blank_updates()
        _set(u, transcription_progress,
             value="Transcription complete!", visible=True)
        _set(u, transcription_tab, visible=True)
        _set(u, transcription_metadata, value=metadata_md)
        _set(u, transcription_display, value=transcription)
        yield u

        if app_state.cancel_requested:
            yield _handle_cancellation(btn_text)
            return

        clear_gpu_memory("after Whisper")

        # ==================================================================
        # STEP 2  –  Sequential minute generation
        # ==================================================================
        logger.info(
            f"Generating minutes for {n_models} model(s) sequentially"
        )

        for i, config in enumerate(saved_configs):
            if i >= MAX_RESULT_SLOTS:
                logger.warning(
                    f"Exceeded max result slots ({MAX_RESULT_SLOTS}), "
                    "skipping remaining configs"
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

            # Unique key for stats (same model + different quant → distinct)
            unique_key = f"{model_id}_{i}"

            # Tab label with quant summary
            qt_short = (
                config["quant_type"] if config["bits"] == "4" else "int8"
            )
            tab_label = f"{display_name} ({config['bits']}-bit {qt_short})"
            stats_display = tab_label

            slot = RESULT_SLOTS[i]

            # -- reveal this slot's tab with "loading" state ----------------
            u = _blank_updates()
            _set(u, slot["tab"], visible=True, label=tab_label)
            _set(u, slot["info"],
                 value=get_model_info_html(model_id, quant_cfg))
            _set(u, slot["status"],
                 value=f"Loading {display_name}...")
            _set(u, slot["progress"],
                 value=progress_html(5, "Loading model..."))
            _set(u, slot["minutes"], value="")
            yield u

            # progress callback
            def _make_cb(idx, name):
                def cb(pct, msg):
                    base = 0.2 + (0.7 / n_models) * idx
                    portion = 0.7 / n_models
                    progress(base + pct * portion, desc=f"{name}: {msg}")
                return cb

            model_cb = _make_cb(i, display_name)
            logger.info(
                f"Processing model {i+1}/{n_models}: "
                f"{model_id} with {quant_cfg}"
            )

            minutes, error, stats = generate_minutes(
                model_id=model_id,
                transcription=transcription,
                quant_config=quant_cfg,
                stats_collector=app_state.stats_collector,
                hf_token=app_state.hf_token,
                progress_callback=model_cb,
                stats_key=unique_key,
                stats_display_name=stats_display,
            )

            clear_gpu_memory(
                f"after model {i+1}/{n_models} - {model_id}"
            )

            u = _blank_updates()
            if error:
                app_state.errors[unique_key] = error
                logger.error(f"Error for {model_id}: {error}")
                _set(u, slot["status"], value="Generation failed.")
                _set(u, slot["progress"],
                     value=progress_html(100, "Error"))
                _set(u, slot["minutes"],
                     value=f"**Error:** {error}")
            else:
                app_state.results[unique_key] = minutes
                logger.info(f"Success for {model_id}")
                _set(u, slot["status"], value="Generation complete!")
                _set(u, slot["progress"],
                     value=progress_html(100, "Complete"))
                _set(u, slot["minutes"], value=minutes)
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
            _set(u, performance_chart,
                 value=create_performance_charts(successful))
            _set(u, memory_chart,
                 value=create_memory_charts(successful))
        else:
            _set(u, stats_tab_col, visible=True)
            _set(u, stats_notice,
                 value="No statistics available \u2014 all models failed.",
                 visible=True)
            _set(u, stats_group, visible=False)
        yield u

        # ==================================================================
        # STEP 4  –  GPT-4o-mini analysis  (only if >= 2 succeeded)
        # ==================================================================
        progress(0.95, desc="Analyzing results...")
        analysis_text = ""
        if len(app_state.results) >= 2 and app_state.openai_key:
            names = {}
            for i, config in enumerate(saved_configs):
                ukey = f"{config['model_id']}_{i}"
                if ukey in app_state.results:
                    qt_s = (
                        config["quant_type"]
                        if config["bits"] == "4" else "int8"
                    )
                    names[ukey] = (
                        f"{config['display_name']} "
                        f"({config['bits']}-bit {qt_s})"
                    )
            analysis_text = analyze_minutes(
                app_state.results, names, app_state.openai_key
            )

        progress(1.0, desc="Complete!")

        u = _blank_updates()
        _set(u, generate_btn, value=btn_text, variant="primary")
        _set(u, transcription_progress, visible=False)
        if analysis_text:
            _set(u, analysis_tab, visible=True)
            _set(u, analysis_output, value=analysis_text)
        else:
            _set(u, analysis_tab, visible=False)
        yield u

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        u = _blank_updates()
        _set(u, error_display,
             value=f"**Unexpected Error:** {str(e)}", visible=True)
        _set(u, generate_btn, value=btn_text, variant="primary")
        yield u
    finally:
        app_state.is_processing = False


def _handle_cancellation(btn_text: str = "Generate Minute"):
    logger.info("Processing cancelled by user")
    clear_gpu_memory()
    clear_transcription_cache()
    u = _blank_updates()
    _set(u, error_display,
         value="**Processing cancelled.** All progress has been lost.",
         visible=True)
    _set(u, generate_btn, value=btn_text, variant="primary")
    _set(u, transcription_progress, visible=False)
    _set(u, transcription_tab, visible=False)
    _set(u, analysis_tab, visible=False)
    for slot in RESULT_SLOTS:
        _set(u, slot["tab"], visible=False)
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
    global transcription_tab, transcription_metadata, transcription_display
    global analysis_tab, analysis_output
    global performance_chart, memory_chart
    global stats_group, stats_notice, stats_tab_col

    defaults = get_default_config()
    model_choices = get_model_choices()

    css = """
    .gradio-container { max-width: 1200px !important; }
    .model-warning    { color: #ffa500; font-size: 0.9em; }
    .disabled-reason  { color: #888; font-style: italic; font-size: 0.85em;
                        margin-top: 4px; }
    .time-warning     { color: #ccc; font-size: 0.85em; margin-top: 6px; }
    /* Tooltip styles */
    .tt-container { margin-bottom: 2px; }
    .tt-icon {
        display: inline-flex; align-items: center; justify-content: center;
        width: 18px; height: 18px; border-radius: 50%;
        background: #4a90d9; color: #fff; font-size: 11px;
        font-weight: 700; font-style: italic; font-family: Georgia, serif;
        cursor: help; flex-shrink: 0;
    }
    .tt-content {
        max-height: 0; overflow: hidden; opacity: 0;
        transition: max-height 0.25s ease, opacity 0.2s ease,
                    padding 0.25s ease, margin 0.25s ease;
        background: #1a1a2e; color: #c8c8d0;
        padding: 0 12px; margin-top: 0;
        border-radius: 6px; font-size: 0.82em; line-height: 1.45;
        border: 1px solid transparent;
    }
    .tt-container:hover .tt-content {
        max-height: 300px; opacity: 1;
        padding: 10px 12px; margin-top: 6px;
        border-color: rgba(255,255,255,0.08);
    }
    /* Spacing above generate button */
    .generate-row { margin-top: 24px !important; }
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
            "using various AI models.  Compare outputs and get AI-powered "
            "analysis."
        )

        # ---- Saved configs state (list of dicts) -------------------------
        saved_configs_state = gr.State([])

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

                # -- model selection & quantization -----------------------
                with gr.Row():
                    # LEFT COLUMN: model + quant + save
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "### Model Selection & Quantization Settings"
                        )

                        model_selector = gr.Dropdown(
                            label="Select Model",
                            choices=model_choices,
                            multiselect=False,
                            info=(
                                "Select a model to configure its "
                                "quantization settings"
                            ),
                        )
                        gr.Markdown(
                            "*\u26a0\ufe0f Some models require access "
                            "approval. Visit the model page on Hugging "
                            "Face to request access before using.*",
                            elem_classes=["model-warning"],
                        )

                        # Bits
                        gr.HTML(_tooltip_html("Bits", TOOLTIP_BITS))
                        bits_dropdown = gr.Dropdown(
                            label="",
                            choices=get_bits_options(),
                            value=defaults["bits"],
                            show_label=False,
                        )

                        # Double Quantization
                        gr.HTML(
                            _tooltip_html(
                                "Double Quantization",
                                TOOLTIP_DOUBLE_QUANT,
                            )
                        )
                        double_quant_dropdown = gr.Dropdown(
                            label="",
                            choices=get_double_quant_options(
                                defaults["bits"]
                            )[0],
                            value=defaults["double_quant"],
                            show_label=False,
                        )
                        double_quant_reason = gr.Markdown(
                            visible=False,
                            elem_classes=["disabled-reason"],
                        )

                        # Quantization Type
                        gr.HTML(
                            _tooltip_html(
                                "Quantization Type", TOOLTIP_QUANT_TYPE
                            )
                        )
                        quant_type_dropdown = gr.Dropdown(
                            label="",
                            choices=get_quant_type_options(
                                defaults["bits"]
                            )[0],
                            value=defaults["quant_type"],
                            show_label=False,
                        )
                        quant_type_reason = gr.Markdown(
                            visible=False,
                            elem_classes=["disabled-reason"],
                        )

                        # Save error + button
                        save_error = gr.Markdown(visible=False)
                        save_btn = gr.Button(
                            "Save Model Quantization Settings",
                            variant="primary",
                        )

                    # RIGHT COLUMN: selected models list
                    with gr.Column(scale=1):
                        gr.Markdown("### Selected Models")
                        selected_models_html = gr.HTML(
                            _render_saved_models([])
                        )
                        remove_dropdown = gr.Dropdown(
                            label="Remove a model",
                            choices=[],
                            multiselect=False,
                            interactive=True,
                            visible=False,
                        )
                        remove_btn = gr.Button(
                            "Remove Selected",
                            variant="secondary",
                            size="sm",
                            visible=False,
                        )
                        time_warning_display = gr.Markdown(
                            "", elem_classes=["time-warning"]
                        )

                # -- generate button (full width) --------------------------
                with gr.Row(elem_classes=["generate-row"]):
                    generate_btn = gr.Button(
                        "Generate Minute", variant="primary", scale=2
                    )

                # -- status bar (below the button during processing) -------
                transcription_progress = gr.Markdown(visible=False)

            # =============================================================
            # TAB: Audio Transcription (hidden until complete)
            # =============================================================
            transcription_tab = gr.Tab(
                "Audio Transcription", visible=False
            )
            with transcription_tab:
                gr.Markdown("## Audio Transcription")
                transcription_metadata = gr.Markdown("")
                transcription_display = gr.Markdown("")

            # =============================================================
            # TABS: result slots (all hidden at start)
            # =============================================================
            RESULT_SLOTS.clear()
            for i in range(MAX_RESULT_SLOTS):
                tab = gr.Tab(f"Model {i + 1}", visible=False)
                with tab:
                    slot_info = gr.Markdown("")
                    slot_status = gr.Markdown("")
                    slot_progress = gr.HTML("")
                    gr.Markdown("### Generated Minutes")
                    slot_minutes = gr.Markdown("")

                RESULT_SLOTS.append({
                    "tab": tab,
                    "info": slot_info,
                    "status": slot_status,
                    "progress": slot_progress,
                    "minutes": slot_minutes,
                })

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
                            performance_chart = gr.Plot(
                                label="Performance Statistics"
                            )
                        with gr.Tab("Memory"):
                            memory_chart = gr.Plot(
                                label="Memory Statistics"
                            )

            # =============================================================
            # TAB: GPT-4o Best Minute Analysis (hidden until done)
            # =============================================================
            analysis_tab = gr.Tab(
                "GPT-4o Best Minute Analysis", visible=False
            )
            with analysis_tab:
                gr.Markdown("## GPT-4o Best Minute Analysis")
                analysis_output = gr.Markdown("")

        # =================================================================
        # Save / Remove handlers
        # =================================================================

        # -- Shared output components for save/remove handlers -------------
        _save_outputs = [
            saved_configs_state,
            selected_models_html,
            remove_dropdown,
            remove_btn,
            generate_btn,
            time_warning_display,
            save_error,
            model_selector,
            bits_dropdown,
            double_quant_dropdown,
            double_quant_reason,
            quant_type_dropdown,
            quant_type_reason,
        ]

        _remove_outputs = [
            saved_configs_state,
            selected_models_html,
            remove_dropdown,
            remove_btn,
            generate_btn,
            time_warning_display,
        ]

        def _time_warning(n: int) -> str:
            if n > 1:
                return (
                    f"*Processing {n} model(s) sequentially may take "
                    f"approximately {n * 3}\u2013{n * 8} minutes depending "
                    f"on model size and audio length.*"
                )
            return ""

        def _on_save(model_id, bits, double_quant, quant_type,
                     saved_configs):
            """Handle 'Save Model Quantization Settings' click."""
            no_change = gr.update()

            if not model_id:
                return (
                    saved_configs,      # state unchanged
                    no_change,          # selected_models_html
                    no_change,          # remove_dropdown
                    no_change,          # remove_btn
                    no_change,          # generate_btn
                    no_change,          # time_warning
                    gr.update(          # save_error
                        value="\u26a0\ufe0f Please select a model first.",
                        visible=True,
                    ),
                    no_change,          # model_selector
                    no_change,          # bits_dropdown
                    no_change,          # double_quant_dropdown
                    no_change,          # double_quant_reason
                    no_change,          # quant_type_dropdown
                    no_change,          # quant_type_reason
                )

            display_name = get_model_display_name(model_id)
            new_config = {
                "model_id": model_id,
                "display_name": display_name,
                "bits": bits,
                "double_quant": double_quant,
                "quant_type": quant_type if bits == "4" else "N/A",
            }

            new_saved = saved_configs + [new_config]
            n = len(new_saved)

            # Reset quant dropdowns to 4-bit defaults
            dq_opts = get_double_quant_options(defaults["bits"])
            qt_opts = get_quant_type_options(defaults["bits"])

            return (
                new_saved,
                gr.update(value=_render_saved_models(new_saved)),
                gr.update(
                    choices=_remove_choices(new_saved),
                    value=None, visible=True,
                ),
                gr.update(visible=True),
                (
                    gr.update(value=_btn_label(n))
                    if not app_state.is_processing
                    else no_change
                ),
                gr.update(value=_time_warning(n)),
                gr.update(visible=False),               # save_error
                gr.update(value=None),                   # model_selector
                gr.update(value=defaults["bits"]),       # bits_dropdown
                gr.update(                               # dq dropdown
                    choices=dq_opts[0],
                    value=defaults["double_quant"],
                    interactive=True,
                ),
                gr.update(visible=False),                # dq reason
                gr.update(                               # qt dropdown
                    choices=qt_opts[0],
                    value=defaults["quant_type"],
                    interactive=True,
                ),
                gr.update(visible=False),                # qt reason
            )

        save_btn.click(
            fn=_on_save,
            inputs=[
                model_selector,
                bits_dropdown,
                double_quant_dropdown,
                quant_type_dropdown,
                saved_configs_state,
            ],
            outputs=_save_outputs,
        )

        def _on_remove(remove_choice, saved_configs):
            """Handle 'Remove Selected' click."""
            no_change = gr.update()

            if not remove_choice or not saved_configs:
                return (
                    saved_configs, no_change, no_change,
                    no_change, no_change, no_change,
                )

            # Extract index from "1. Model Name (...)"
            try:
                idx = int(remove_choice.split(".")[0]) - 1
            except (ValueError, IndexError):
                return (
                    saved_configs, no_change, no_change,
                    no_change, no_change, no_change,
                )

            if 0 <= idx < len(saved_configs):
                new_saved = saved_configs[:idx] + saved_configs[idx + 1:]
            else:
                new_saved = saved_configs

            n = len(new_saved)
            show_remove = n > 0

            return (
                new_saved,
                gr.update(value=_render_saved_models(new_saved)),
                gr.update(
                    choices=_remove_choices(new_saved),
                    value=None, visible=show_remove,
                ),
                gr.update(visible=show_remove),
                (
                    gr.update(value=_btn_label(n))
                    if not app_state.is_processing
                    else no_change
                ),
                gr.update(value=_time_warning(n)),
            )

        remove_btn.click(
            fn=_on_remove,
            inputs=[remove_dropdown, saved_configs_state],
            outputs=_remove_outputs,
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
            transcription_tab,      # 3
            transcription_metadata, # 4
            transcription_display,  # 5
            analysis_tab,           # 6
            analysis_output,        # 7
            stats_tab_col,          # 8
            stats_notice,           # 9
            stats_group,            # 10
            performance_chart,      # 11
            memory_chart,           # 12
        ])
        for slot in RESULT_SLOTS:
            OUTPUT_COMPONENTS.append(slot["tab"])
            OUTPUT_COMPONENTS.append(slot["info"])
            OUTPUT_COMPONENTS.append(slot["status"])
            OUTPUT_COMPONENTS.append(slot["progress"])
            OUTPUT_COMPONENTS.append(slot["minutes"])

        OUTPUT_INDEX.clear()
        for idx, comp in enumerate(OUTPUT_COMPONENTS):
            OUTPUT_INDEX[id(comp)] = idx

        # -- generate button -> handler ----------------------------------
        generate_btn.click(
            fn=on_generate_click,
            inputs=[audio_input, saved_configs_state],
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
