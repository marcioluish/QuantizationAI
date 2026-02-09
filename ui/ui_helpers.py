"""
UI rendering helpers and CSS for the Meeting Minutes Generator.
Pure functions for HTML/markdown and output-update helpers.
"""

from typing import Dict, List, Optional

import gradio as gr

from ui.app_state import OUTPUT_COMPONENTS, OUTPUT_INDEX
from engine.model_configs import get_model_info


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

APP_CSS = """
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


# ---------------------------------------------------------------------------
# Output update helpers
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


# ---------------------------------------------------------------------------
# Tooltip and saved-models HTML
# ---------------------------------------------------------------------------

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
# Model info markdown (for result tabs)
# ---------------------------------------------------------------------------

def get_model_info_html(
    model_id: str, quant_cfg: Optional[Dict[str, str]] = None
) -> str:
    """Build markdown for model information and optional quantization config."""
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
