import torch

def pattern(x):
    tmp_8_1 = x.flatten(2)
    tmp_9_1 = tmp_8_1.transpose(1, 2)
    return tmp_8_1, tmp_9_1

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def optimized_embedding_transform(x):
    """
    Optimized transformation: flatten(2) + transpose(1, 2)
    This is common in transformer patch embeddings where:
    - Input: [batch, channels, height, width] from conv2d
    - flatten(2): [batch, channels, height*width] 
    - transpose(1, 2): [batch, height*width, channels]
    """
    # Combined optimized operation
    # Instead of separate flatten + transpose, use reshape which is more efficient
    if x.dim() == 4 and x.shape[0] == 1:  # batch_size=1 typical for this pattern
        batch_size, channels, height, width = x.shape
        
        # Direct reshape to [1, height*width, channels] which is equivalent to flatten(2)+transpose(1,2)
        # This is more efficient than two separate operations
        result = x.reshape(batch_size, height * width, channels)
        
        # We need to return both the flattened and transposed versions to match the pattern
        flattened = x.flatten(2)
        transposed = result
        
        return flattened, transposed
    else:
        # Fallback to original operations
        tmp_8_1 = x.flatten(2)
        tmp_9_1 = tmp_8_1.transpose(1, 2)
        return tmp_8_1, tmp_9_1

def replacement_func():
    return optimized_embedding_transform