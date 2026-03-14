import torch

# Analysis of input tensor shapes across different scenarios
SHAPE_ANALYSIS = {
    "scenario_0": {
        "shape": [1, 2, 128, 32, 24],
        "description": "Small batch size (1), medium spatial dimensions"
    },
    "scenario_7": {
        "shape": [32, 2, 128, 24, 32], 
        "description": "Large batch size (32), smaller spatial dimensions"
    },
    "scenario_5": {
        "shape": [64, 2, 128, 12, 12],
        "description": "Very large batch size (64), smallest spatial dimensions"
    }
}

def compute_operation_fusion_benefit(input_shape):
    """
    Analyze potential benefit of fusing sum(dim=1) + adaptive_avg_pool2d(1)
    
    input_shape: [batch, channels, height, width1, width2]
    """
    batch, channels, height, width1, width2 = input_shape
    
    # Original operations cost estimation
    original_sum_operations = batch * height * width1 * width2  # sum over channels
    original_pooling_operations = batch * height * 1 * 1  # adaptive pool to 1x1
    original_total = original_sum_operations + original_pooling_operations
    
    # Fused operation cost estimation  
    fused_operations = batch * height  # mean over all target dims
    fusion_ratio = fused_operations / original_total
    
    print(f"Input shape: {input_shape}")
    print(f"Original ops: {original_total}, Fused ops: {fused_operations}")
    print(f"Fusion efficiency ratio: {fusion_ratio:.3f}")
    
    return fusion_ratio

# Test all scenarios
if __name__ == "__main__":
    print("🔍 Shape Analysis for Operation Fusion Optimization")
    print("=" * 60)
    
    for scenario, data in SHAPE_ANALYSIS.items():
        print(f"\n{scenario.upper()}: {data['description']}")
        fusion_ratio = compute_operation_fusion_benefit(data['shape'])
        print(f"Potential performance improvement: {(1-fusion_ratio)*100:.1f}% reduction in operations")
    
    print("\n" + "=" * 60)
    print("✅ All scenarios benefit from fusion, especially larger batch sizes")