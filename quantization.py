"""
Quantization configurations for the Meeting Minutes Generator.
Handles BitsAndBytes configuration options and validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


@dataclass
class QuantizationOption:
    """Represents a quantization option with its constraints."""
    value: str
    display_name: str
    enabled: bool = True
    disabled_reason: Optional[str] = None


# Available options for each quantization parameter
BITS_OPTIONS = ["4", "8"]
DOUBLE_QUANT_OPTIONS = ["True", "False"]
QUANT_TYPE_OPTIONS = ["nf4", "fp4"]


def get_bits_options() -> List[str]:
    """Get available bits options."""
    return BITS_OPTIONS.copy()


def get_double_quant_options(bits: str) -> Tuple[List[str], bool, Optional[str]]:
    """
    Get double quantization options based on selected bits.
    Returns: (options, enabled, disabled_reason)
    """
    if bits == "8":
        return (
            ["False"],
            False,
            "8-bit quantization does not support double quantization. "
            "Double quantization is only available with 4-bit quantization."
        )
    return (DOUBLE_QUANT_OPTIONS.copy(), True, None)


def get_quant_type_options(bits: str) -> Tuple[List[str], bool, Optional[str]]:
    """
    Get quantization type options based on selected bits.
    Returns: (options, enabled, disabled_reason)
    """
    if bits == "8":
        return (
            ["N/A"],
            False,
            "8-bit quantization uses a fixed quantization scheme. "
            "Quantization type selection (nf4/fp4) is only available with 4-bit quantization."
        )
    return (QUANT_TYPE_OPTIONS.copy(), True, None)


def create_quantization_config(
    bits: str,
    double_quant: str,
    quant_type: str
) -> Optional[BitsAndBytesConfig]:
    """
    Create a BitsAndBytesConfig based on user selections.
    
    Args:
        bits: "4" or "8"
        double_quant: "True" or "False"
        quant_type: "nf4" or "fp4"
    
    Returns:
        BitsAndBytesConfig object or None if BitsAndBytes not available
    """
    if BitsAndBytesConfig is None:
        raise ImportError("BitsAndBytes is not installed")
    
    if bits == "8":
        return BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:  # 4-bit
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=(double_quant == "True"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=quant_type
        )


def get_quantization_summary(bits: str, double_quant: str, quant_type: str) -> str:
    """Get a human-readable summary of the quantization configuration."""
    if bits == "8":
        return "8-bit quantization"
    else:
        dq_str = "with double quantization" if double_quant == "True" else "without double quantization"
        return f"4-bit {quant_type.upper()} quantization {dq_str}"


def get_default_config() -> Dict[str, str]:
    """Get default quantization configuration."""
    return {
        "bits": "4",
        "double_quant": "True",
        "quant_type": "nf4"
    }
