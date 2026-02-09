"""
Meeting minutes generation module using LLM models.
Handles model loading, generation, and progress tracking.
"""

from typing import Callable, Optional, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from engine.model_configs import get_model_info, get_model_display_name
from engine.quantization import create_quantization_config
from engine.stats import StatisticsCollector, ModelStatistics
from engine.utils import clear_gpu_memory, log_gpu_memory, filter_thinking_tokens, check_model_access, logger


# System prompt for minute generation
SYSTEM_PROMPT = """You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks."""

# User prompt template
USER_PROMPT_TEMPLATE = """Below is a transcript of a meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription:
{transcription}"""


class FirstTokenRecorder(StoppingCriteria):
    """Records time to first token using a stopping criteria hook."""
    
    def __init__(self, callback: Callable[[], None]):
        self.callback = callback
        self.recorded = False
    
    def __call__(self, input_ids, scores, **kwargs):
        if not self.recorded:
            self.callback()
            self.recorded = True
        return False


def generate_minutes(
    model_id: str,
    transcription: str,
    quant_config: Dict[str, str],
    stats_collector: StatisticsCollector,
    hf_token: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    stats_key: Optional[str] = None,
    stats_display_name: Optional[str] = None,
) -> Tuple[str, Optional[str], ModelStatistics]:
    """
    Generate meeting minutes using a specified model.
    
    Args:
        model_id: The Hugging Face model ID
        transcription: The transcribed text from the audio
        quant_config: Quantization configuration dict
        stats_collector: Statistics collector instance
        hf_token: Hugging Face token for gated models
        progress_callback: Optional callback for progress updates
        stats_key: Unique key for statistics (defaults to model_id)
        stats_display_name: Display name for statistics charts
        
    Returns:
        Tuple of (generated_minutes, error_message, statistics)
    """
    model_info = get_model_info(model_id)
    display_name = stats_display_name or get_model_display_name(model_id)
    _stats_key = stats_key or model_id
    
    # Initialize statistics
    stats = stats_collector.start_model(_stats_key, display_name)
    stats_collector.reset_gpu_peak_memory()
    
    model = None
    tokenizer = None
    
    try:
        # Progress: 0%
        if progress_callback:
            progress_callback(0.0, "Checking model access...")
        
        logger.info(f"Starting minute generation with {model_id}")
        
        # Check model access
        has_access, access_error = check_model_access(model_id, hf_token)
        if not has_access:
            stats_collector.finish_model(_stats_key, 0, success=False, error_message=access_error)
            return "", access_error, stats
        
        # Progress: 10%
        if progress_callback:
            progress_callback(0.10, "Loading tokenizer...")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {model_id}")
        log_gpu_memory(f"before tokenizer load - {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Progress: 25%
        if progress_callback:
            progress_callback(0.25, "Loading model with quantization...")
        
        # Create quantization config
        logger.info(f"Creating quantization config: {quant_config}")
        bnb_config = create_quantization_config(
            bits=quant_config["bits"],
            double_quant=quant_config["double_quant"],
            quant_type=quant_config["quant_type"]
        )
        
        # Load model
        logger.info(f"Loading model {model_id}")
        log_gpu_memory(f"before model load - {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
            token=hf_token,
            trust_remote_code=True
        )
        log_gpu_memory(f"after model load - {model_id}")
        
        # Progress: 50%
        if progress_callback:
            progress_callback(0.50, "Preparing input...")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcription=transcription)}
        ]
        
        # Tokenize input – apply_chat_template may return a plain tensor
        # or a BatchEncoding depending on transformers version.
        logger.info("Tokenizing input")
        tokenized = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        # Extract tensor: BatchEncoding → .input_ids, plain tensor → use directly
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized["input_ids"].to(model.device)
        else:
            input_ids = tokenized.to(model.device)
        
        input_length = input_ids.shape[1]
        logger.info(f"Input length: {input_length} tokens")
        
        # Progress: 60%
        if progress_callback:
            progress_callback(0.60, "Generating minutes...")
        
        # Generate
        logger.info("Starting generation")
        log_gpu_memory(f"before generate - {model_id}")
        with torch.no_grad():
            # Track first token timing via stopping criteria
            first_token_criteria = FirstTokenRecorder(
                lambda: stats_collector.record_first_token(_stats_key)
            )
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([first_token_criteria])
            )
        
        # Progress: 90%
        if progress_callback:
            progress_callback(0.90, "Processing output...")
        
        log_gpu_memory(f"after generate - {model_id}")
        
        # Decode output
        generated_tokens = outputs[0][input_length:]
        tokens_generated = len(generated_tokens)
        
        minutes = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Filter thinking tokens for distill models
        if model_info and model_info.requires_thinking_filter:
            logger.info("Filtering thinking tokens from output")
            minutes = filter_thinking_tokens(minutes)
        
        # Finish statistics
        stats_collector.finish_model(_stats_key, tokens_generated, success=True)
        
        # Clean up model
        logger.info("Cleaning up model")
        del model
        del tokenizer
        model = None
        tokenizer = None
        clear_gpu_memory()
        
        # Progress: 100%
        if progress_callback:
            progress_callback(1.0, "Generation complete!")
        
        logger.info(f"Generation complete for {model_id}. Output length: {len(minutes)} chars")
        
        return minutes, None, stats
        
    except Exception as e:
        error_msg = f"Generation failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Explicitly release model/tokenizer so GPU memory is freed
        if model is not None:
            del model
            model = None
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
        clear_gpu_memory(f"error cleanup - {model_id}")
        
        # Record failure
        stats_collector.finish_model(_stats_key, 0, success=False, error_message=error_msg)
        
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        
        return "", error_msg, stats
