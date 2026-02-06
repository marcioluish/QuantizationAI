"""
Utility functions for the Meeting Minutes Generator.
"""

import gc
import os
import re
import logging
from typing import Optional, Tuple

import torch

# Configure logging - use force=True to override any prior basicConfig
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger("meeting_minutes")
logger.setLevel(logging.INFO)


def log_gpu_memory(label: str = "") -> None:
    """Log current GPU memory usage. Call at every key step."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - reserved
    logger.info(
        f"[GPU {label}] "
        f"Allocated: {allocated:.2f} GB | "
        f"Reserved: {reserved:.2f} GB | "
        f"Free: {free:.2f} GB | "
        f"Total: {total:.2f} GB"
    )


def clear_gpu_memory(label: str = ""):
    """Clear GPU memory and run garbage collection."""
    log_gpu_memory(f"before cleanup{' - ' + label if label else ''}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log_gpu_memory(f"after cleanup{' - ' + label if label else ''}")


def get_gpu_memory_info() -> Tuple[float, float, float]:
    """
    Get GPU memory information.
    
    Returns:
        Tuple of (total_gb, used_gb, free_gb)
    """
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0)
    
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    free = total - reserved
    
    return (total, allocated, free)


def validate_audio_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an uploaded audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file_path:
        return False, "No file uploaded. Please upload an MP3 file."
    
    # Check file extension
    if not file_path.lower().endswith('.mp3'):
        return False, "Invalid file type. Please upload an MP3 file."
    
    # Check file exists
    if not os.path.exists(file_path):
        return False, "File not found. Please try uploading again."
    
    # Check file size (max 50MB)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 50:
        return False, f"File too large ({file_size_mb:.1f}MB). Maximum size is 50MB."
    
    logger.info(f"Audio file validated: {file_path} ({file_size_mb:.1f}MB)")
    return True, None


def validate_model_selection(selected_models: list) -> Tuple[bool, Optional[str]]:
    """
    Validate model selection.
    
    Args:
        selected_models: List of selected model IDs
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not selected_models:
        return False, "No models selected. Please select at least one model."
    
    logger.info(f"Model selection validated: {selected_models}")
    return True, None


def check_model_access(model_id: str, hf_token: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Check if user has access to a model on Hugging Face.
    
    Args:
        model_id: The model ID to check
        hf_token: Optional Hugging Face token
        
    Returns:
        Tuple of (has_access, error_message)
    """
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=hf_token)
        info = api.model_info(model_id, token=hf_token)
        
        # If the model is gated and we got here, we have access.
        # If we did NOT have access the call above would have raised
        # a RepositoryNotFoundError / GatedRepoAccessError.
        logger.info(f"Model access verified: {model_id}")
        return True, None
        
    except Exception as e:
        err_str = str(e).lower()
        if "gated" in err_str or "access" in err_str or "403" in err_str:
            error_msg = (
                f"Access denied to {model_id}. This is a gated model. "
                f"Please visit https://huggingface.co/{model_id} to request access, "
                f"then ensure your HF_TOKEN has the required permissions."
            )
            logger.warning(f"Model access denied: {model_id}")
        else:
            error_msg = f"Failed to check model access for {model_id}: {str(e)}"
            logger.error(error_msg)
        return False, error_msg


def filter_thinking_tokens(text: str) -> str:
    """
    Filter out thinking/reasoning tokens from DeepSeek distill models.
    Removes content between <think> and </think> tags.
    
    Args:
        text: The generated text
        
    Returns:
        Text with thinking tokens removed
    """
    # Remove <think>...</think> blocks
    filtered = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Also handle potential variations
    filtered = re.sub(r'<thinking>.*?</thinking>', '', filtered, flags=re.DOTALL)
    filtered = re.sub(r'\[think\].*?\[/think\]', '', filtered, flags=re.DOTALL)
    
    # Clean up extra whitespace
    filtered = re.sub(r'\n{3,}', '\n\n', filtered)
    filtered = filtered.strip()
    
    return filtered


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_memory(bytes_val: float) -> str:
    """Format bytes into a human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val:.0f} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/(1024**2):.1f} MB"
    else:
        return f"{bytes_val/(1024**3):.2f} GB"
