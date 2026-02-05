"""
Model configurations and metadata for the Meeting Minutes Generator.
Contains all available models with their properties and descriptions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about a model for display and loading."""
    model_id: str
    display_name: str
    organization: str
    parameters: str
    size_fp16: str
    size_4bit: str
    is_fine_tuned: bool
    fine_tune_method: Optional[str]
    description: str
    is_distill: bool = False
    requires_thinking_filter: bool = False


# All available models sorted alphabetically by organization
AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": ModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        display_name="DeepSeek R1 Distill Llama 8B",
        organization="DeepSeek AI",
        parameters="8B",
        size_fp16="~16GB",
        size_4bit="~5GB",
        is_fine_tuned=True,
        fine_tune_method="Knowledge Distillation from DeepSeek-R1",
        description="A distilled version of DeepSeek-R1 based on Llama architecture. "
                    "Optimized for reasoning tasks while being more efficient than the full model.",
        is_distill=True,
        requires_thinking_filter=True
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": ModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        display_name="DeepSeek R1 Distill Qwen 1.5B",
        organization="DeepSeek AI",
        parameters="1.5B",
        size_fp16="~3GB",
        size_4bit="~1GB",
        is_fine_tuned=True,
        fine_tune_method="Knowledge Distillation from DeepSeek-R1",
        description="Smallest distilled version of DeepSeek-R1 based on Qwen architecture. "
                    "Very fast and memory efficient while maintaining good reasoning capabilities.",
        is_distill=True,
        requires_thinking_filter=True
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ModelInfo(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        display_name="DeepSeek R1 Distill Qwen 7B",
        organization="DeepSeek AI",
        parameters="7B",
        size_fp16="~14GB",
        size_4bit="~4GB",
        is_fine_tuned=True,
        fine_tune_method="Knowledge Distillation from DeepSeek-R1",
        description="Distilled version of DeepSeek-R1 based on Qwen architecture. "
                    "Balances performance and efficiency for reasoning tasks.",
        is_distill=True,
        requires_thinking_filter=True
    ),
    "meta-llama/Llama-3.1-8B-Instruct": ModelInfo(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama 3.1 8B Instruct",
        organization="Meta",
        parameters="8B",
        size_fp16="~16GB",
        size_4bit="~5GB",
        is_fine_tuned=True,
        fine_tune_method="Instruction Fine-Tuning (SFT + RLHF)",
        description="Meta's instruction-tuned Llama 3.1 model. Excellent at following "
                    "instructions and generating structured content like meeting minutes."
    ),
    "meta-llama/Llama-3.2-3B-Instruct": ModelInfo(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        display_name="Llama 3.2 3B Instruct",
        organization="Meta",
        parameters="3B",
        size_fp16="~6GB",
        size_4bit="~2GB",
        is_fine_tuned=True,
        fine_tune_method="Instruction Fine-Tuning (SFT + RLHF)",
        description="Compact instruction-tuned model from Meta's Llama 3.2 family. "
                    "Fast and efficient while maintaining good instruction-following capabilities."
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": ModelInfo(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        display_name="Mistral 7B Instruct v0.3",
        organization="Mistral AI",
        parameters="7B",
        size_fp16="~14GB",
        size_4bit="~4GB",
        is_fine_tuned=True,
        fine_tune_method="Instruction Fine-Tuning",
        description="Mistral AI's instruction-tuned model known for excellent performance "
                    "relative to its size. Great balance of speed and quality."
    ),
    "Qwen/Qwen2.5-3B-Instruct": ModelInfo(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        display_name="Qwen 2.5 3B Instruct",
        organization="Alibaba (Qwen)",
        parameters="3B",
        size_fp16="~6GB",
        size_4bit="~2GB",
        is_fine_tuned=True,
        fine_tune_method="Instruction Fine-Tuning (SFT + RLHF)",
        description="Alibaba's compact instruction model from the Qwen 2.5 family. "
                    "Efficient and capable for structured text generation tasks."
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelInfo(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen 2.5 7B Instruct",
        organization="Alibaba (Qwen)",
        parameters="7B",
        size_fp16="~14GB",
        size_4bit="~4GB",
        is_fine_tuned=True,
        fine_tune_method="Instruction Fine-Tuning (SFT + RLHF)",
        description="Alibaba's instruction-tuned Qwen 2.5 model. Strong multilingual "
                    "capabilities and excellent at following complex instructions."
    ),
}


def get_model_choices() -> List[str]:
    """Get list of model IDs sorted alphabetically by organization."""
    return sorted(
        AVAILABLE_MODELS.keys(),
        key=lambda x: (AVAILABLE_MODELS[x].organization, AVAILABLE_MODELS[x].display_name)
    )


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get model info by ID."""
    return AVAILABLE_MODELS.get(model_id)


def get_model_summary(model_id: str) -> str:
    """Generate a formatted summary for model tooltip/info display."""
    info = get_model_info(model_id)
    if not info:
        return "Unknown model"
    
    fine_tune_info = f"Fine-tuned: {info.fine_tune_method}" if info.is_fine_tuned else "Fine-tuned: No"
    
    return f"""**{info.display_name}**
    
**Organization:** {info.organization}
**Parameters:** {info.parameters}
**Size (FP16):** {info.size_fp16}
**Size (4-bit):** {info.size_4bit}
**{fine_tune_info}**

{info.description}"""


def get_model_display_name(model_id: str) -> str:
    """Get short display name for tabs."""
    info = get_model_info(model_id)
    return info.display_name if info else model_id.split("/")[-1]
