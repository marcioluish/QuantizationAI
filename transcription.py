"""
Audio transcription module using OpenAI Whisper.
Handles transcription of audio files with progress reporting.
"""

import logging
from typing import Callable, Optional, Tuple

import torch

from utils import clear_gpu_memory, log_gpu_memory, logger


# Transcription cache (in-memory)
_transcription_cache = {}


def transcribe_audio(
    audio_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[str, Optional[str]]:
    """
    Transcribe an audio file using Whisper medium.en model.
    
    Args:
        audio_path: Path to the audio file
        progress_callback: Optional callback for progress updates (progress_pct, status_msg)
        
    Returns:
        Tuple of (transcription, error_message)
    """
    # Check cache first
    if audio_path in _transcription_cache:
        logger.info(f"Using cached transcription for {audio_path}")
        if progress_callback:
            progress_callback(1.0, "Using cached transcription")
        return _transcription_cache[audio_path], None
    
    try:
        # Progress: 0%
        if progress_callback:
            progress_callback(0.0, "Loading Whisper model...")
        
        logger.info("Loading Whisper medium.en model...")
        log_gpu_memory("before Whisper load")
        
        from transformers import pipeline
        
        # Load the pipeline (float16 to halve GPU memory usage)
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-medium.en",
            dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu",
            return_timestamps=True
        )
        log_gpu_memory("after Whisper load")
        
        # Progress: 25%
        if progress_callback:
            progress_callback(0.25, "Whisper model loaded. Starting transcription...")
        
        logger.info(f"Starting transcription of {audio_path}")
        
        # Progress: 50%
        if progress_callback:
            progress_callback(0.50, "Transcribing audio...")
        
        log_gpu_memory("before Whisper inference")
        result = pipe(audio_path)
        log_gpu_memory("after Whisper inference")
        
        # Progress: 75%
        if progress_callback:
            progress_callback(0.75, "Processing transcription results...")
        
        transcription = result.get("text", "")
        
        if not transcription:
            logger.warning("Transcription returned empty result")
            return "", "Transcription returned empty result. Please check the audio file."
        
        # Cache the transcription
        _transcription_cache[audio_path] = transcription
        
        # Progress: 100%
        if progress_callback:
            progress_callback(1.0, "Transcription complete!")
        
        logger.info(f"Transcription complete. Length: {len(transcription)} characters")
        
        # Unload the model to free GPU memory
        del pipe
        clear_gpu_memory()
        
        return transcription, None
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Clean up on error
        clear_gpu_memory()
        
        return "", error_msg


def clear_transcription_cache():
    """Clear the transcription cache."""
    global _transcription_cache
    _transcription_cache = {}
    logger.info("Transcription cache cleared")
