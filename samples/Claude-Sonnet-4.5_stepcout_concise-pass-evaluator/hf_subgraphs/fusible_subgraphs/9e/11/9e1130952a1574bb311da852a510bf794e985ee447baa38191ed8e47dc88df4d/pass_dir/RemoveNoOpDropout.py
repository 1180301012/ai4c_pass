import torch
import triton
import triton.language as tl

# Pattern: Match just the dropout(p=0.0) + dropout(p=0.0) pattern and eliminate it
# This is a standalone pattern that removes two consecutive no-op dropouts
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_8, tmp_7

# Extract arguments
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

# Optimized implementation: Remove dropout (no-op with p=0.0)
# This eliminates two dropout kernel launches
def replacement_func():
    def optimized_path(in_0, in_1, in_2, in_3, in_4, in_5):
        # Compute add + mean
        tmp_4 = in_5 + in_4
        tmp_5 = tmp_4.mean((2, 3), keepdim=False)
        
        # Skip dropout (p=0.0 is a no-op), directly use tmp_5
        # But we still need to return both outputs
        # tmp_7 in original was dropout(dropout(tmp_5)) = tmp_5 (since dropout is no-op)
        
        # Batch norm
        tmp_8 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
        
        # Return (batch_norm_output, mean_output) - same as original return structure
        return tmp_8, tmp_5
    return optimized_path