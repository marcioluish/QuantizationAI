"""
Shared application state and registries for the Meeting Minutes Generator.
Holds AppState singleton, component registries, and token initialization.
"""

import os
from typing import Dict, List, Optional

import gradio as gr

from engine.stats import StatisticsCollector
from engine.utils import logger


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
        self.errors: Dict[str, str] = {}   # unique_key -> error message


app_state = AppState()

# Registries populated by create_app() and consumed by event handlers
RESULT_SLOTS: List[Dict[str, gr.Component]] = []
OUTPUT_COMPONENTS: List[gr.Component] = []
OUTPUT_INDEX: Dict[int, int] = {}   # id(component) -> index


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
