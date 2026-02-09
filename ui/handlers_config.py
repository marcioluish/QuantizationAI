"""
Event handlers for the Configuration tab: save, remove, quantization dropdowns.
"""

import gradio as gr

from ui.app_state import app_state
from engine.model_configs import get_model_display_name
from engine.quantization import (
    get_double_quant_options,
    get_quant_type_options,
    get_default_config,
)
from ui.ui_helpers import (
    _btn_label,
    _remove_choices,
    _render_saved_models,
)


def update_double_quant_options(bits: str):
    """Cascading dropdown: bits -> double quant options."""
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
    """Cascading dropdown: bits -> quant type options."""
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


def _time_warning(n: int) -> str:
    if n > 1:
        return (
            f"*Processing {n} model(s) sequentially may take "
            f"approximately {n * 3}\u2013{n * 8} minutes depending "
            f"on model size and audio length.*"
        )
    return ""


def wire_config_events(config, saved_configs_state, defaults):
    """
    Connect save button, remove button, and bits dropdown to their handlers.
    config: dict from build_config_tab().
    """
    no_change = gr.update()

    def on_save(model_id, bits, double_quant, quant_type, saved_configs):
        if not model_id:
            return (
                saved_configs,
                no_change,
                no_change,
                no_change,
                no_change,
                no_change,
                gr.update(
                    value="\u26a0\ufe0f Please select a model first.",
                    visible=True,
                ),
                no_change,
                no_change,
                no_change,
                no_change,
                no_change,
                no_change,
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
        dq_opts = get_double_quant_options(defaults["bits"])
        qt_opts = get_quant_type_options(defaults["bits"])

        return (
            new_saved,
            gr.update(value=_render_saved_models(new_saved)),
            gr.update(
                choices=_remove_choices(new_saved),
                value=None,
                visible=True,
            ),
            gr.update(visible=True),
            (
                gr.update(value=_btn_label(n))
                if not app_state.is_processing
                else no_change
            ),
            gr.update(value=_time_warning(n)),
            gr.update(visible=False),
            gr.update(value=None),
            gr.update(value=defaults["bits"]),
            gr.update(
                choices=dq_opts[0],
                value=defaults["double_quant"],
                interactive=True,
            ),
            gr.update(visible=False),
            gr.update(
                choices=qt_opts[0],
                value=defaults["quant_type"],
                interactive=True,
            ),
            gr.update(visible=False),
        )

    save_outputs = [
        saved_configs_state,
        config["selected_models_html"],
        config["remove_dropdown"],
        config["remove_btn"],
        config["generate_btn"],
        config["time_warning_display"],
        config["save_error"],
        config["model_selector"],
        config["bits_dropdown"],
        config["double_quant_dropdown"],
        config["double_quant_reason"],
        config["quant_type_dropdown"],
        config["quant_type_reason"],
    ]
    config["save_btn"].click(
        fn=on_save,
        inputs=[
            config["model_selector"],
            config["bits_dropdown"],
            config["double_quant_dropdown"],
            config["quant_type_dropdown"],
            saved_configs_state,
        ],
        outputs=save_outputs,
    )

    def on_remove(remove_choice, saved_configs):
        if not remove_choice or not saved_configs:
            return (
                saved_configs,
                no_change,
                no_change,
                no_change,
                no_change,
                no_change,
            )
        try:
            idx = int(remove_choice.split(".")[0]) - 1
        except (ValueError, IndexError):
            return (
                saved_configs,
                no_change,
                no_change,
                no_change,
                no_change,
                no_change,
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
                value=None,
                visible=show_remove,
            ),
            gr.update(visible=show_remove),
            (
                gr.update(value=_btn_label(n))
                if not app_state.is_processing
                else no_change
            ),
            gr.update(value=_time_warning(n)),
        )

    remove_outputs = [
        saved_configs_state,
        config["selected_models_html"],
        config["remove_dropdown"],
        config["remove_btn"],
        config["generate_btn"],
        config["time_warning_display"],
    ]
    config["remove_btn"].click(
        fn=on_remove,
        inputs=[config["remove_dropdown"], saved_configs_state],
        outputs=remove_outputs,
    )

    config["bits_dropdown"].change(
        fn=update_double_quant_options,
        inputs=[config["bits_dropdown"]],
        outputs=[config["double_quant_dropdown"], config["double_quant_reason"]],
    ).then(
        fn=update_quant_type_options,
        inputs=[config["bits_dropdown"]],
        outputs=[config["quant_type_dropdown"], config["quant_type_reason"]],
    )
