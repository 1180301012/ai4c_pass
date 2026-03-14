import torch

def pattern(x):
    tmp_13 = x.view(1, 96, 96, 128)
    tmp_14 = torch.nn.functional.pad(tmp_13, (0, 0, 0, 0, 0, 0), 'constant', None)
    tmp_15 = tmp_14.view(1, 8, 12, 8, 12, 128)
    tmp_16 = tmp_15.permute(0, 1, 3, 2, 4, 5)
    tmp_17 = tmp_16.contiguous()
    tmp_18 = tmp_17.view(-1, 12, 12, 128)
    tmp_19 = tmp_18.view(-1, 144, 128)
    return tmp_13, tmp_19

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def optimized_final_reshape(x):
    """
    Optimized final reshape sequence. This sequence appears in transformer window attention:
    
    Original sequence:
    1. view(1, 96, 96, 128)
    2. pad(no-op, all zeros)
    3. view(1, 8, 12, 8, 12, 128) 
    4. permute(0, 1, 3, 2, 4, 5)
    5. contiguous()
    6. view(-1, 12, 12, 128)
    7. view(-1, 144, 128)
    
    This reshapes the tensor for window attention operations.
    """
    batch_size = x.shape[0]
    
    # Optimized version that skips unnecessary operations
    # Step 1: view(1, 96, 96, 128) - this is needed for intermediate output
    tmp_13 = x.view(batch_size, 96, 96, 128)
    
    # Skip padding - it's a no-op with all zeros
    
    # Direct transformation from [1, 96, 96, 128] to final [1, 9216, 144]
    # This skips all the intermediate reshape operations
    if batch_size == 1:
        # For the specific case: [1, 96, 96, 128] → [1, 9216, 128] → [1, 9216, 144]
        # Note: 96*96 = 9216, and 12*12 = 144
        # The transform breaks down the spatial dimensions for windows
        
        # First reshape to [1, 8, 12, 8, 12, 128] equivalent
        tmp_intermediate = x.view(batch_size, 8, 12, 8, 12, 128)
        
        # Permute [0,1,3,2,4,5] 
        tmp_permuted = tmp_intermediate.permute(0, 1, 3, 2, 4, 5)
        
        # Skip contiguous() if memory is already contiguous (optimization)
        
        # Final reshape operations: [1, 8, 8, 12, 12, 128] → [64*144, 128] → [9216, 144, 128]
        tmp_final = tmp_permuted.reshape(-1, 12, 12, 128)
        tmp_19 = tmp_final.reshape(-1, 144, 128)
        
        return tmp_13, tmp_19
    else:
        # Fallback to original operations for unsupported cases
        tmp_13 = x.view(1, 96, 96, 128)
        tmp_14 = torch.nn.functional.pad(tmp_13, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_15 = tmp_14.view(1, 8, 12, 8, 12, 128)
        tmp_16 = tmp_15.permute(0, 1, 3, 2, 4, 5)
        tmp_17 = tmp_16.contiguous()
        tmp_18 = tmp_17.view(-1, 12, 12, 128)
        tmp_19 = tmp_18.view(-1, 144, 128)
        return tmp_13, tmp_19

def replacement_func():
    return optimized_final_reshape