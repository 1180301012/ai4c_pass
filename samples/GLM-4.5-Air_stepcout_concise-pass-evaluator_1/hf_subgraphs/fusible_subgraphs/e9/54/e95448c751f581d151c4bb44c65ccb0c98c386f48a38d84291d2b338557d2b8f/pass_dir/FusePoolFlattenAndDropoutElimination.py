import torch

def pattern(feature_map, gelu_out):
    # Adaptive average pooling 2D reduces to 1x1
    pooled = torch.nn.functional.adaptive_avg_pool2d(gelu_out, 1)
    # Flatten the 1x1 result to 1D
    flattened = pooled.flatten(1, -1)
    # Dropout with p=0.0 is a no-op, returns input unchanged
    dropout_out = torch.nn.functional.dropout(flattened, 0.0, False, False)
    return pooled, flattened, dropout_out

def replacement_args(feature_map, gelu_out):
    return (feature_map, gelu_out)

def replacement_func():
    def optimized_pool_flatten_dropout(feature_map, gelu_out):
        # For adaptive_avg_pool2d with target size=1, we can compute mean directly
        batch_size, channels, height, width = gelu_out.shape
        
        # Compute spatial mean manually (equivalent to adaptive_avg_pool2d(1))
        spatial_mean = torch.mean(gelu_out, dim=[2, 3], keepdim=False)
        
        # This directly gives us the flattened result since pooling to 1x1 then flatten
        # is equivalent to spatial mean across H,W dimensions
        
        # Dropout with p=0.0 is just identity, so return spatial_mean directly
        return spatial_mean
    
    return optimized_pool_flatten_dropout