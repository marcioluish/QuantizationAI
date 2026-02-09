"""
Gradio tab construction for the Meeting Minutes Generator.
Each function builds one tab (or slot group) and returns a dict of components.
"""

import gradio as gr

from ui.app_state import MAX_RESULT_SLOTS, RESULT_SLOTS
from engine.quantization import (
    get_bits_options,
    get_double_quant_options,
    get_quant_type_options,
    TOOLTIP_BITS,
    TOOLTIP_DOUBLE_QUANT,
    TOOLTIP_QUANT_TYPE,
)
from ui.ui_helpers import _render_saved_models, _tooltip_html


def build_config_tab(model_choices, defaults):
    """Build the Configuration tab. Returns dict of all components."""
    with gr.Tab("Configuration", id="config_tab"):
        error_display = gr.Markdown(visible=False)

        with gr.Row():
            audio_input = gr.Audio(
                label="Upload Audio File (MP3, max 50MB)",
                type="filepath",
                sources=["upload"],
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Selection & Quantization Settings")

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

                gr.HTML(_tooltip_html("Bits", TOOLTIP_BITS))
                bits_dropdown = gr.Dropdown(
                    label="",
                    choices=get_bits_options(),
                    value=defaults["bits"],
                    show_label=False,
                )

                gr.HTML(_tooltip_html("Double Quantization", TOOLTIP_DOUBLE_QUANT))
                double_quant_dropdown = gr.Dropdown(
                    label="",
                    choices=get_double_quant_options(defaults["bits"])[0],
                    value=defaults["double_quant"],
                    show_label=False,
                )
                double_quant_reason = gr.Markdown(
                    visible=False,
                    elem_classes=["disabled-reason"],
                )

                gr.HTML(_tooltip_html("Quantization Type", TOOLTIP_QUANT_TYPE))
                quant_type_dropdown = gr.Dropdown(
                    label="",
                    choices=get_quant_type_options(defaults["bits"])[0],
                    value=defaults["quant_type"],
                    show_label=False,
                )
                quant_type_reason = gr.Markdown(
                    visible=False,
                    elem_classes=["disabled-reason"],
                )

                save_error = gr.Markdown(visible=False)
                save_btn = gr.Button(
                    "Save Model Quantization Settings",
                    variant="primary",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Selected Models")
                selected_models_html = gr.HTML(_render_saved_models([]))
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
                    "",
                    elem_classes=["time-warning"],
                )

        with gr.Row(elem_classes=["generate-row"]):
            generate_btn = gr.Button(
                "Generate Minute",
                variant="primary",
                scale=2,
            )

        transcription_progress = gr.Markdown(visible=False)

    return {
        "error_display": error_display,
        "audio_input": audio_input,
        "model_selector": model_selector,
        "bits_dropdown": bits_dropdown,
        "double_quant_dropdown": double_quant_dropdown,
        "double_quant_reason": double_quant_reason,
        "quant_type_dropdown": quant_type_dropdown,
        "quant_type_reason": quant_type_reason,
        "save_error": save_error,
        "save_btn": save_btn,
        "selected_models_html": selected_models_html,
        "remove_dropdown": remove_dropdown,
        "remove_btn": remove_btn,
        "time_warning_display": time_warning_display,
        "generate_btn": generate_btn,
        "transcription_progress": transcription_progress,
    }


def build_transcription_tab():
    """Build the Audio Transcription tab (hidden until complete)."""
    tab = gr.Tab("Audio Transcription", visible=False)
    with tab:
        gr.Markdown("## Audio Transcription")
        metadata = gr.Markdown("")
        display = gr.Markdown("")
    return {"tab": tab, "metadata": metadata, "display": display}


def build_result_slots(n):
    """Build n result slot tabs. Clears and populates RESULT_SLOTS. Returns RESULT_SLOTS."""
    RESULT_SLOTS.clear()
    for i in range(n):
        t = gr.Tab(f"Model {i + 1}", visible=False)
        with t:
            slot_info = gr.Markdown("")
            slot_status = gr.Markdown("")
            slot_progress = gr.HTML("")
            gr.Markdown("### Generated Minutes")
            slot_minutes = gr.Markdown("")
        RESULT_SLOTS.append({
            "tab": t,
            "info": slot_info,
            "status": slot_status,
            "progress": slot_progress,
            "minutes": slot_minutes,
        })
    return RESULT_SLOTS


def build_statistics_tab():
    """Build the Statistics tab (hidden until results ready)."""
    tab = gr.Tab("Statistics", visible=False)
    with tab:
        notice = gr.Markdown(visible=False)
        group = gr.Group(visible=False)
        with group:
            with gr.Tabs():
                with gr.Tab("Performance"):
                    performance_chart = gr.Plot(label="Performance Statistics")
                with gr.Tab("Memory"):
                    memory_chart = gr.Plot(label="Memory Statistics")
    return {
        "tab": tab,
        "notice": notice,
        "group": group,
        "performance_chart": performance_chart,
        "memory_chart": memory_chart,
    }


def build_analysis_tab():
    """Build the GPT-4o Best Minute Analysis tab (hidden until done)."""
    tab = gr.Tab("GPT-4o Best Minute Analysis", visible=False)
    with tab:
        gr.Markdown("## GPT-4o Best Minute Analysis")
        output = gr.Markdown("")
    return {"tab": tab, "output": output}
