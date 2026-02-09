"""
Meeting Minutes Generator - Gradio Application
Slim orchestrator: builds UI from tab modules and wires events.
"""

import gradio as gr

from ui.app_state import (
    MAX_RESULT_SLOTS,
    OUTPUT_COMPONENTS,
    OUTPUT_INDEX,
    RESULT_SLOTS,
    initialize_tokens,
)
from ui.ui_helpers import APP_CSS
from ui.ui_tabs import (
    build_analysis_tab,
    build_config_tab,
    build_result_slots,
    build_statistics_tab,
    build_transcription_tab,
)
from ui.handlers_config import wire_config_events
from ui.handlers_generate import on_generate_click, register_components

# Token init on import
initialize_tokens()


def create_app():
    """Build and return the Gradio Blocks application."""
    from engine.model_configs import get_model_choices
    from engine.quantization import get_default_config

    defaults = get_default_config()
    model_choices = get_model_choices()

    with gr.Blocks(
        title="Meeting Minutes Generator",
        theme=gr.themes.Soft(primary_hue="blue").set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
        ),
        css=APP_CSS,
    ) as app:

        gr.Markdown(
            "# Meeting Minutes Generator\n\n"
            "Upload an audio file and generate professional meeting minutes "
            "using various AI models.  Compare outputs and get AI-powered "
            "analysis."
        )
        saved_configs_state = gr.State([])

        with gr.Tabs():
            config = build_config_tab(model_choices, defaults)
            trans = build_transcription_tab()
            build_result_slots(MAX_RESULT_SLOTS)
            stats = build_statistics_tab()
            analysis = build_analysis_tab()

        wire_config_events(config, saved_configs_state, defaults)

        OUTPUT_COMPONENTS.clear()
        OUTPUT_COMPONENTS.extend([
            config["error_display"],
            config["generate_btn"],
            config["transcription_progress"],
            trans["tab"],
            trans["metadata"],
            trans["display"],
            analysis["tab"],
            analysis["output"],
            stats["tab"],
            stats["notice"],
            stats["group"],
            stats["performance_chart"],
            stats["memory_chart"],
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

        register_components({
            "error_display": config["error_display"],
            "generate_btn": config["generate_btn"],
            "transcription_progress": config["transcription_progress"],
            "transcription_tab": trans["tab"],
            "transcription_metadata": trans["metadata"],
            "transcription_display": trans["display"],
            "analysis_tab": analysis["tab"],
            "analysis_output": analysis["output"],
            "stats_tab_col": stats["tab"],
            "stats_notice": stats["notice"],
            "stats_group": stats["group"],
            "performance_chart": stats["performance_chart"],
            "memory_chart": stats["memory_chart"],
        })

        config["generate_btn"].click(
            fn=on_generate_click,
            inputs=[config["audio_input"], saved_configs_state],
            outputs=OUTPUT_COMPONENTS,
        )

    return app


def launch_app():
    app = create_app()
    app.launch(share=True, debug=True)


if __name__ == "__main__":
    launch_app()
