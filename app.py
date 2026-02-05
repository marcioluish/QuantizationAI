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
    logger
)


# Global state
class AppState:
    """Global application state."""
    def __init__(self):
        self.is_processing = False
        self.cancel_requested = False
        self.hf_token: Optional[str] = None
        self.openai_key: Optional[str] = None
        self.stats_collector: Optional[StatisticsCollector] = None
        self.results: Dict[str, str] = {}  # model_id -> generated minutes
        self.errors: Dict[str, str] = {}   # model_id -> error message


app_state = AppState()

# UI component registries
MODEL_COMPONENTS: Dict[str, Dict[str, gr.Component]] = {}
MODEL_ORDER: List[str] = []
OUTPUT_COMPONENTS: List[gr.Component] = []
OUTPUT_INDEX: Dict[gr.Component, int] = {}


def progress_html(percent: int, status: str) -> str:
    """Return HTML for a simple progress bar with status."""
    percent = max(0, min(100, int(percent)))
    return (
        f"<div style='margin-bottom:6px;'>"
        f"<progress value='{percent}' max='100' style='width:100%;'></progress>"
        f"</div><div>{status}</div>"
    )


def _blank_updates():
    """Create a list of empty updates for all outputs."""
    return [gr.update() for _ in OUTPUT_COMPONENTS]


def _set_update(updates, component: gr.Component, **kwargs) -> None:
    """Set a component update in the updates list."""
    index = OUTPUT_INDEX.get(component)
    if index is None:
        return
    updates[index] = gr.update(**kwargs)


def initialize_tokens():
    """Initialize API tokens from environment/Colab secrets."""
    try:
        # Try Colab secrets first
        from google.colab import userdata
        app_state.hf_token = userdata.get('HF_TOKEN')
        app_state.openai_key = userdata.get('OPENAI_API_KEY')
        logger.info("Loaded tokens from Colab secrets")
    except Exception:
        # Fall back to environment variables
        app_state.hf_token = os.environ.get('HF_TOKEN')
        app_state.openai_key = os.environ.get('OPENAI_API_KEY')
        logger.info("Loaded tokens from environment variables")
    
    # Login to HuggingFace if token available
    if app_state.hf_token:
        try:
            from huggingface_hub import login
            login(app_state.hf_token, add_to_git_credential=False)
            logger.info("Logged in to Hugging Face")
        except Exception as e:
            logger.warning(f"Failed to login to Hugging Face: {e}")


def update_double_quant_options(bits: str) -> Tuple[gr.Dropdown, gr.Markdown]:
    """Update double quantization options based on bits selection."""
    options, enabled, reason = get_double_quant_options(bits)
    
    if enabled:
        return (
            gr.Dropdown(choices=options, value=options[0], interactive=True),
            gr.Markdown(visible=False)
        )
    else:
        return (
            gr.Dropdown(choices=options, value=options[0], interactive=False),
            gr.Markdown(value=f"*{reason}*", visible=True)
        )


def update_quant_type_options(bits: str) -> Tuple[gr.Dropdown, gr.Markdown]:
    """Update quantization type options based on bits selection."""
    options, enabled, reason = get_quant_type_options(bits)
    
    if enabled:
        return (
            gr.Dropdown(choices=options, value=options[0], interactive=True),
            gr.Markdown(visible=False)
        )
    else:
        return (
            gr.Dropdown(choices=options, value=options[0], interactive=False),
            gr.Markdown(value=f"*{reason}*", visible=True)
        )


def validate_inputs(
    audio_file: Optional[str],
    selected_models: List[str]
) -> Tuple[bool, str]:
    """Validate all inputs before processing."""
    # Validate audio file
    is_valid, error = validate_audio_file(audio_file)
    if not is_valid:
        return False, error
    
    # Validate model selection
    is_valid, error = validate_model_selection(selected_models)
    if not is_valid:
        return False, error
    
    return True, ""


def on_generate_click(
    audio_file: Optional[str],
    selected_models: List[str],
    bits: str,
    double_quant: str,
    quant_type: str,
    progress=gr.Progress()
):
    """
    Main handler for the Generate Minutes button.
    Orchestrates transcription and minute generation.
    """
    # Validate inputs
    is_valid, error = validate_inputs(audio_file, selected_models)
    if not is_valid:
        updates = _blank_updates()
        _set_update(updates, error_display, value=f"**Error:** {error}", visible=True)
        _set_update(updates, generate_btn, value="Generate Minutes", variant="primary")
        yield updates
        return
    
    # Start processing
    app_state.is_processing = True
    app_state.cancel_requested = False
    app_state.stats_collector = StatisticsCollector()
    app_state.results = {}
    app_state.errors = {}
    
    try:
        # Update button to Cancel and initialize model tabs
        updates = _blank_updates()
        _set_update(updates, error_display, visible=False)
        _set_update(updates, generate_btn, value="Cancel", variant="stop")
        _set_update(updates, transcription_progress, value="Starting...", visible=True)
        _set_update(updates, transcription_accordion, visible=False)
        _set_update(updates, analysis_output, visible=False)
        
        # Initialize model tabs
        for model_id in MODEL_ORDER:
            display_name = get_model_display_name(model_id)
            status = (
                f"Model {display_name} minute processing didn't start yet."
                if model_id in selected_models
                else f"Model {display_name} not selected."
            )
            _set_update(updates, MODEL_COMPONENTS[model_id]["status"], value=status)
            _set_update(updates, MODEL_COMPONENTS[model_id]["progress"], value=progress_html(0, "Not started"))
            _set_update(updates, MODEL_COMPONENTS[model_id]["minutes"], value="")
        
        yield updates
        
        # Check for cancellation
        if app_state.cancel_requested:
            yield _handle_cancellation()
            return
        
        # Step 1: Transcription
        logger.info("Starting transcription")
        
        def transcription_progress_callback(pct, msg):
            progress(pct * 0.3, desc=msg)  # 0-30% for transcription
        
        transcription, trans_error = transcribe_audio(
            audio_file,
            progress_callback=transcription_progress_callback
        )
        
        if trans_error:
            updates = _blank_updates()
            _set_update(updates, error_display, value=f"**Transcription Error:** {trans_error}", visible=True)
            _set_update(updates, generate_btn, value="Generate Minutes", variant="primary")
            _set_update(updates, transcription_progress, visible=False)
            yield updates
            return
        
        # Show transcription
        updates = _blank_updates()
        _set_update(updates, transcription_progress, value="Transcription complete!", visible=True)
        _set_update(
            updates,
            transcription_accordion,
            label="Transcription (click to expand)",
            open=False,
            visible=True
        )
        _set_update(updates, transcription_display, value=transcription)
        yield updates
        
        # Check for cancellation
        if app_state.cancel_requested:
            yield _handle_cancellation()
            return
        
        # Prepare quantization config
        quant_config = {
            "bits": bits,
            "double_quant": double_quant,
            "quant_type": quant_type
        }
        
        # Step 2: Generate minutes for each model
        logger.info(f"Generating minutes for {len(selected_models)} models")
        
        # Clear GPU before loading minute generation models
        clear_gpu_memory()
        
        # Process models sequentially (GPU memory constraint)
        for i, model_id in enumerate(selected_models):
            if app_state.cancel_requested:
                yield _handle_cancellation()
                return
            
            display_name = get_model_display_name(model_id)
            updates = _blank_updates()
            _set_update(
                updates,
                MODEL_COMPONENTS[model_id]["status"],
                value="Loading model and preparing input..."
            )
            _set_update(
                updates,
                MODEL_COMPONENTS[model_id]["progress"],
                value=progress_html(10, "Loading model...")
            )
            yield updates
            
            def model_progress_callback(pct, msg):
                # 30-90% for model generation, split among models
                base = 0.3 + (0.6 / len(selected_models)) * i
                model_portion = 0.6 / len(selected_models)
                progress(base + pct * model_portion, desc=f"{display_name}: {msg}")
            
            logger.info(f"Processing model {i+1}/{len(selected_models)}: {model_id}")
            
            minutes, error, stats = generate_minutes(
                model_id=model_id,
                transcription=transcription,
                quant_config=quant_config,
                stats_collector=app_state.stats_collector,
                hf_token=app_state.hf_token,
                progress_callback=model_progress_callback
            )
            
            if error:
                app_state.errors[model_id] = error
                logger.error(f"Error for {model_id}: {error}")
                updates = _blank_updates()
                _set_update(
                    updates,
                    MODEL_COMPONENTS[model_id]["status"],
                    value="Generation failed."
                )
                _set_update(
                    updates,
                    MODEL_COMPONENTS[model_id]["progress"],
                    value=progress_html(100, "Error")
                )
                _set_update(
                    updates,
                    MODEL_COMPONENTS[model_id]["minutes"],
                    value=error
                )
                yield updates
            else:
                app_state.results[model_id] = minutes
                logger.info(f"Success for {model_id}")
                updates = _blank_updates()
                _set_update(
                    updates,
                    MODEL_COMPONENTS[model_id]["status"],
                    value="Generation complete!"
                )
                _set_update(
                    updates,
                    MODEL_COMPONENTS[model_id]["progress"],
                    value=progress_html(100, "Complete")
                )
                _set_update(
                    updates,
                    MODEL_COMPONENTS[model_id]["minutes"],
                    value=minutes
                )
                yield updates
        
        # Step 3: GPT-4o-mini analysis (if 2+ models succeeded)
        progress(0.95, desc="Analyzing results...")
        
        analysis_text = ""
        if len(app_state.results) >= 2 and app_state.openai_key:
            model_display_names = {
                mid: get_model_display_name(mid) 
                for mid in app_state.results.keys()
            }
            analysis_text = analyze_minutes(
                app_state.results,
                model_display_names,
                app_state.openai_key
            )
        
        # Update stats charts if any model succeeded
        stats_updates = _blank_updates()
        successful_stats = app_state.stats_collector.get_successful_stats()
        if successful_stats:
            perf_fig = create_performance_charts(successful_stats)
            mem_fig = create_memory_charts(successful_stats)
            _set_update(stats_updates, performance_chart, value=perf_fig)
            _set_update(stats_updates, memory_chart, value=mem_fig)
            _set_update(stats_updates, stats_group, visible=True)
            _set_update(stats_updates, stats_notice, visible=False)
        else:
            _set_update(stats_updates, stats_group, visible=False)
            _set_update(
                stats_updates,
                stats_notice,
                value="No statistics available because all models failed.",
                visible=True
            )
        yield stats_updates
        
        # Build final output
        progress(1.0, desc="Complete!")
        
        # Prepare updates
        updates = _blank_updates()
        _set_update(updates, generate_btn, value="Generate Minutes", variant="primary")
        _set_update(updates, transcription_progress, visible=False)
        
        # Add analysis if available
        if analysis_text:
            _set_update(updates, analysis_output, value=analysis_text, visible=True)
        else:
            _set_update(updates, analysis_output, visible=False)
        
        yield updates
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        updates = _blank_updates()
        _set_update(updates, error_display, value=f"**Unexpected Error:** {str(e)}", visible=True)
        _set_update(updates, generate_btn, value="Generate Minutes", variant="primary")
        yield updates
    finally:
        app_state.is_processing = False


def _handle_cancellation():
    """Handle cancellation cleanup and UI updates."""
    logger.info("Processing cancelled by user")
    clear_gpu_memory()
    clear_transcription_cache()
    
    updates = _blank_updates()
    _set_update(updates, error_display, value="**Processing cancelled.** All progress has been lost.", visible=True)
    _set_update(updates, generate_btn, value="Generate Minutes", variant="primary")
    _set_update(updates, transcription_progress, visible=False)
    return updates


def on_cancel_click():
    """Handle cancel button click."""
    if app_state.is_processing:
        # Show confirmation (handled via JS in real implementation)
        app_state.cancel_requested = True
        return gr.Button(value="Cancelling...", interactive=False)
    return gr.Button(value="Generate Minutes", variant="primary")


def get_model_info_html(model_id: str) -> str:
    """Generate HTML for model info display."""
    info = get_model_info(model_id)
    if not info:
        return ""
    
    return f"""
### Model Information

| Property | Value |
|----------|-------|
| **Organization** | {info.organization} |
| **Parameters** | {info.parameters} |
| **Size (FP16)** | {info.size_fp16} |
| **Size (4-bit)** | {info.size_4bit} |
| **Fine-tuned** | {"Yes - " + info.fine_tune_method if info.is_fine_tuned else "No"} |

{info.description}
"""


# Initialize tokens on import
initialize_tokens()


def create_app():
    """Create and return the Gradio application."""
    
    # Define component references (will be assigned in the UI)
    global error_display, generate_btn, transcription_progress
    global transcription_accordion, transcription_display, analysis_output
    global performance_chart, memory_chart, stats_group, stats_notice
    
    # Get default values
    defaults = get_default_config()
    model_choices = get_model_choices()
    
    # Custom CSS for dark theme
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .model-warning {
        color: #ffa500;
        font-size: 0.9em;
    }
    .disabled-reason {
        color: #888;
        font-style: italic;
        font-size: 0.85em;
        margin-top: 4px;
    }
    """
    
    with gr.Blocks(
        title="Meeting Minutes Generator",
        theme=gr.themes.Soft(primary_hue="blue").set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
        ),
        css=custom_css
    ) as app:
        
        gr.Markdown("""
        # Meeting Minutes Generator
        
        Upload an audio file and generate professional meeting minutes using various AI models.
        Compare outputs from different models and get AI-powered analysis of the results.
        """)
        
        with gr.Tabs() as main_tabs:
            
            # Main Configuration Tab
            with gr.Tab("Configuration", id="config_tab"):
                
                # Error display
                error_display = gr.Markdown(visible=False, elem_classes=["error-message"])
                
                # Row 1: File Upload
                with gr.Row():
                    audio_input = gr.Audio(
                        label="Upload Audio File (MP3, max 50MB)",
                        type="filepath",
                        sources=["upload"]
                    )
                
                # Row 2: Model Selection and Quantization
                with gr.Row():
                    # Left column: Model selection
                    with gr.Column(scale=1):
                        gr.Markdown("### Model Selection")
                        
                        model_selector = gr.Dropdown(
                            label="Select Models (max 2)",
                            choices=model_choices,
                            multiselect=True,
                            max_choices=2,
                            info="Select up to 2 models to generate minutes"
                        )
                        
                        gr.Markdown(
                            "*⚠️ Some models require access approval. Visit the model page on "
                            "Hugging Face to request access before using.*",
                            elem_classes=["model-warning"]
                        )
                        
                        selected_models_display = gr.Markdown("")
                    
                    # Right column: Quantization options
                    with gr.Column(scale=1):
                        gr.Markdown("### Quantization Settings")
                        
                        bits_dropdown = gr.Dropdown(
                            label="Bits",
                            choices=get_bits_options(),
                            value=defaults["bits"],
                            info="Number of bits for quantization"
                        )
                        
                        double_quant_dropdown = gr.Dropdown(
                            label="Double Quantization",
                            choices=get_double_quant_options(defaults["bits"])[0],
                            value=defaults["double_quant"],
                            info="Apply double quantization for more compression"
                        )
                        double_quant_reason = gr.Markdown(visible=False, elem_classes=["disabled-reason"])
                        
                        quant_type_dropdown = gr.Dropdown(
                            label="Quantization Type",
                            choices=get_quant_type_options(defaults["bits"])[0],
                            value=defaults["quant_type"],
                            info="Type of 4-bit quantization"
                        )
                        quant_type_reason = gr.Markdown(visible=False, elem_classes=["disabled-reason"])
                
                # Row 3: Generate Button
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate Minutes",
                        variant="primary",
                        scale=2
                    )
                
                # Row 4: Progress display
                with gr.Row():
                    transcription_progress = gr.Markdown(visible=False)
                
                # Row 5: Transcription display (collapsed)
                transcription_accordion = gr.Accordion(
                    label="Transcription (click to expand)",
                    open=False,
                    visible=False
                )
                with transcription_accordion:
                    transcription_display = gr.Markdown()
                
                # Row 6: GPT-4o-mini Analysis
                gr.Markdown("### AI Analysis")
                analysis_output = gr.Markdown(
                    visible=False,
                    label="GPT-4o-mini Analysis"
                )
            
            # Model Result Tabs (pre-created for all models)
            MODEL_COMPONENTS.clear()
            MODEL_ORDER.clear()
            for model_id in model_choices:
                display_name = get_model_display_name(model_id)
                MODEL_ORDER.append(model_id)
                with gr.Tab(display_name):
                    gr.Markdown(f"## {display_name}")
                    model_info_md = gr.Markdown(get_model_info_html(model_id))
                    model_status_md = gr.Markdown(
                        value=f"Model {display_name} minute processing didn't start yet."
                    )
                    model_progress_html = gr.HTML(progress_html(0, "Not started"))
                    gr.Markdown("### Generated Minutes")
                    model_minutes_md = gr.Markdown("")
                    
                    MODEL_COMPONENTS[model_id] = {
                        "info": model_info_md,
                        "status": model_status_md,
                        "progress": model_progress_html,
                        "minutes": model_minutes_md
                    }
            
            # Statistics Tab (created after processing)
            with gr.Tab("Statistics", id="stats_tab") as stats_tab:
                stats_notice = gr.Markdown(
                    value="Statistics will appear here after processing completes.",
                    visible=True
                )
                stats_group = gr.Group(visible=False)
                with stats_group:
                    with gr.Tabs():
                        with gr.Tab("Performance"):
                            performance_chart = gr.Plot(label="Performance Statistics")
                        
                        with gr.Tab("Memory"):
                            memory_chart = gr.Plot(label="Memory Statistics")
        
        # Event handlers
        
        # Update model display when selection changes
        def update_model_display(models):
            if not models:
                return ""
            return "**Selected:** " + ", ".join([
                get_model_display_name(m) for m in models
            ])
        
        model_selector.change(
            fn=update_model_display,
            inputs=[model_selector],
            outputs=[selected_models_display]
        )
        
        # Update quantization options when bits change
        bits_dropdown.change(
            fn=update_double_quant_options,
            inputs=[bits_dropdown],
            outputs=[double_quant_dropdown, double_quant_reason]
        ).then(
            fn=update_quant_type_options,
            inputs=[bits_dropdown],
            outputs=[quant_type_dropdown, quant_type_reason]
        )
        
        # Generate button handler
        # Build output list for updates
        OUTPUT_COMPONENTS.clear()
        OUTPUT_COMPONENTS.extend([
            error_display,
            generate_btn,
            transcription_progress,
            transcription_accordion,
            transcription_display,
            analysis_output,
            stats_notice,
            stats_group,
            performance_chart,
            memory_chart
        ])
        
        # Add model components to outputs
        for model_id in MODEL_ORDER:
            OUTPUT_COMPONENTS.extend([
                MODEL_COMPONENTS[model_id]["status"],
                MODEL_COMPONENTS[model_id]["progress"],
                MODEL_COMPONENTS[model_id]["minutes"]
            ])
        
        # Build output index for updates
        OUTPUT_INDEX.clear()
        for idx, comp in enumerate(OUTPUT_COMPONENTS):
            OUTPUT_INDEX[comp] = idx
        
        generate_btn.click(
            fn=on_generate_click,
            inputs=[
                audio_input,
                model_selector,
                bits_dropdown,
                double_quant_dropdown,
                quant_type_dropdown
            ],
            outputs=OUTPUT_COMPONENTS
        )
    
    return app


def launch_app():
    """Launch the Gradio application."""
    app = create_app()
    app.launch(
        share=True,  # Create public link for Colab
        debug=True
    )


if __name__ == "__main__":
    launch_app()
