"""
Statistics collection and visualization for the Meeting Minutes Generator.
Handles performance metrics, memory monitoring, and chart generation.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import psutil
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import torch

from utils import logger


@dataclass
class ModelStatistics:
    """Statistics for a single model's generation."""
    model_id: str
    display_name: str
    
    # Performance stats
    total_time: float = 0.0
    time_to_first_token: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    
    # Memory stats (in bytes)
    gpu_memory_peak: float = 0.0
    gpu_memory_allocated: float = 0.0
    ram_peak: float = 0.0
    
    # Status
    success: bool = False
    error_message: Optional[str] = None
    
    # Color for charts
    color: str = "#1f77b4"


# Color palette for charts
CHART_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
]


class StatisticsCollector:
    """Collects and manages statistics for all models."""
    
    def __init__(self):
        self.model_stats: Dict[str, ModelStatistics] = {}
        self._color_index = 0
        self._start_time: Optional[float] = None
        self._initial_gpu_memory: float = 0.0
        self._process = psutil.Process()
        
    def start_model(self, model_id: str, display_name: str) -> ModelStatistics:
        """Start tracking a model."""
        color = CHART_COLORS[self._color_index % len(CHART_COLORS)]
        self._color_index += 1
        
        stats = ModelStatistics(
            model_id=model_id,
            display_name=display_name,
            color=color
        )
        self.model_stats[model_id] = stats
        
        # Record initial state
        self._start_time = time.time()
        self._initial_gpu_memory = self._get_gpu_memory_allocated()
        
        logger.info(f"Started tracking statistics for {display_name}")
        return stats
    
    def record_first_token(self, model_id: str):
        """Record when the first token was generated."""
        if model_id in self.model_stats and self._start_time:
            self.model_stats[model_id].time_to_first_token = (
                time.time() - self._start_time
            )
            logger.info(
                f"First token for {model_id}: "
                f"{self.model_stats[model_id].time_to_first_token:.2f}s"
            )
    
    def finish_model(
        self, 
        model_id: str, 
        tokens_generated: int,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Finish tracking a model and record final statistics."""
        if model_id not in self.model_stats:
            return
        
        stats = self.model_stats[model_id]
        end_time = time.time()
        
        # Calculate performance metrics
        stats.total_time = end_time - self._start_time if self._start_time else 0.0
        stats.tokens_generated = tokens_generated
        
        if stats.total_time > 0 and tokens_generated > 0:
            stats.tokens_per_second = tokens_generated / stats.total_time
        
        # Memory metrics
        stats.gpu_memory_peak = self._get_gpu_memory_peak()
        stats.gpu_memory_allocated = self._get_gpu_memory_allocated() - self._initial_gpu_memory
        stats.ram_peak = self._process.memory_info().rss
        
        # Status
        stats.success = success
        stats.error_message = error_message
        
        logger.info(
            f"Finished tracking {model_id}: "
            f"time={stats.total_time:.2f}s, "
            f"tokens={tokens_generated}, "
            f"tokens/s={stats.tokens_per_second:.2f}"
        )
    
    def _get_gpu_memory_allocated(self) -> float:
        """Get current GPU memory allocated in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0)
        return 0.0
    
    def _get_gpu_memory_peak(self) -> float:
        """Get peak GPU memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0)
        return 0.0
    
    def reset_gpu_peak_memory(self):
        """Reset the peak memory tracker."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
    
    def get_successful_stats(self) -> List[ModelStatistics]:
        """Get statistics for models that completed successfully."""
        return [s for s in self.model_stats.values() if s.success]
    
    def get_all_stats(self) -> List[ModelStatistics]:
        """Get all model statistics."""
        return list(self.model_stats.values())


def create_performance_charts(stats_list: List[ModelStatistics]) -> go.Figure:
    """
    Create bar charts for performance statistics.
    
    Args:
        stats_list: List of ModelStatistics for successful models
        
    Returns:
        Plotly figure with performance charts
    """
    if not stats_list:
        return _create_empty_chart("No performance data available")
    
    # Create subplots for each metric
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Total Generation Time (seconds)",
            "Tokens per Second",
            "Time to First Token (seconds)"
        ),
        horizontal_spacing=0.1
    )
    
    model_names = [s.display_name for s in stats_list]
    colors = [s.color for s in stats_list]
    
    # Total generation time
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=[s.total_time for s in stats_list],
            marker_color=colors,
            text=[f"{s.total_time:.1f}s" for s in stats_list],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Tokens per second
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=[s.tokens_per_second for s in stats_list],
            marker_color=colors,
            text=[f"{s.tokens_per_second:.1f}" for s in stats_list],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Time to first token
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=[s.time_to_first_token for s in stats_list],
            marker_color=colors,
            text=[f"{s.time_to_first_token:.2f}s" for s in stats_list],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title_text="Performance Statistics",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig


def create_memory_charts(stats_list: List[ModelStatistics]) -> go.Figure:
    """
    Create bar charts for memory statistics.
    
    Args:
        stats_list: List of ModelStatistics for successful models
        
    Returns:
        Plotly figure with memory charts
    """
    if not stats_list:
        return _create_empty_chart("No memory data available")
    
    # Create subplots for each metric
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "GPU Memory Peak (GB)",
            "GPU Memory Allocated (GB)",
            "RAM Peak (GB)"
        ),
        horizontal_spacing=0.1
    )
    
    model_names = [s.display_name for s in stats_list]
    colors = [s.color for s in stats_list]
    
    # Convert bytes to GB for display
    gpu_peak_gb = [s.gpu_memory_peak / (1024**3) for s in stats_list]
    gpu_alloc_gb = [s.gpu_memory_allocated / (1024**3) for s in stats_list]
    ram_peak_gb = [s.ram_peak / (1024**3) for s in stats_list]
    
    # GPU memory peak
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=gpu_peak_gb,
            marker_color=colors,
            text=[f"{v:.2f} GB" for v in gpu_peak_gb],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # GPU memory allocated
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=gpu_alloc_gb,
            marker_color=colors,
            text=[f"{v:.2f} GB" for v in gpu_alloc_gb],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # RAM peak
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=ram_peak_gb,
            marker_color=colors,
            text=[f"{v:.2f} GB" for v in ram_peak_gb],
            textposition='auto',
            showlegend=False
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title_text="Memory Statistics",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    
    return fig


def _create_empty_chart(message: str) -> go.Figure:
    """Create an empty chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        template="plotly_dark",
        height=400
    )
    return fig
