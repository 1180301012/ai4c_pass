"""
AI4C Optimization Pass Package

This package contains optimization passes for deep learning workloads,
specifically targeting attention mechanism patterns and Conv2D operations.

Author: AI4C Optimization Team
Date: 2024
"""

__version__ = "1.0.0"
__author__ = "AI4C Optimization Team"

# Available optimization passes
AVAILABLE_PASSES = [
    "SimpleConv2DTest",
    "FullConvSoftmaxAttention", 
    "SimpleSoftmaxTest",
    "Conv2DViewTest"
]

def get_available_passes():
    """Get list of available optimization passes."""
    return AVAILABLE_PASSES

def get_working_passes():
    """Get list of passes that have been tested and work."""
    return ["SimpleConv2DTest"]  # Only currently working pass

def get_pattern_constraints():
    """Document key constraints discovered during development."""
    return {
        "pattern_matching": {
            "single_operations": "✅ WORKS",
            "multi_operations": "❌ FAILS - Framework constraints",
            "strict_syntax": "Must match exact operation sequence"
        },
        "performance": {
            "conv2d_baseline": "Already optimized by cuDNN",
            "optimization_challenge": "Naive Triton kernels compete poorly",
            "fusion_opportunity": "Multi-operation sequences have potential"
        }
    }

__all__ = ["AVAILABLE_PASSES", "get_available_passes", "get_working_passes", "get_pattern_constraints"]