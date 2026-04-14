import torch
import triton
import triton.language as tl

def pattern(in_x1, ln_weight1, ln_bias1, in_x2, ln_weight2, ln_bias2):
    # First layer norm + sigmoid
    ln_out1 = torch.nn.functional.layer_norm(in_x1, (in_x1.shape[-1],), ln_weight1, ln_bias1, 1e-05)
    sigmoid_out1 = torch.sigmoid(ln_out1)
    
    # Second layer norm
    ln_out2 = torch.nn.functional.layer_norm(in_x2, (in_x2.shape[-1],), ln_weight2, ln_bias2, 1e-05)
    unsqueeze_out = ln_out2.unsqueeze(-2)
    
    # Return all outputs that might be observable
    return sigmoid_out1, unsqueeze_out

def replacement_args(in_x1, ln_weight1, ln_bias1, in_x2, ln_weight2, ln_bias2):
    return (in_x1, ln_weight1, ln_bias1, in_x2, ln_weight2, ln_bias2)

@triton.jit
def layer_norm_sigmoid_unsqueeze_kernel(
    x1_ptr, ln_weight1_ptr, ln_bias1_ptr,
    x2_ptr, ln_weight2_ptr, ln_bias2_ptr,
    sigmoid_out_ptr, unsqueeze_out_ptr,
    x1_elements, x2_elements, hidden_size,
    UNSQUEEZE_DIM: tl.constexpr = -2,
    eps: tl.constexpr = 1e-05
):
    # Each program handles a column of data
    pid = tl.program_id(0)
    n_cols = tl.cdiv(x1_elements, hidden_size)
    
    if pid >= n_cols:
        return
    
    col_base = pid * hidden_size
    
    # Load weights and biases
    ln_weight1 = tl.load(ln_weight1_ptr + pid)
    ln_bias1 = tl.load(ln_bias1_ptr + pid)
    ln_weight2 = tl.load(ln_weight2_ptr + pid)
    ln_bias2 = tl.load(ln_bias2_ptr + pid)
    
    # Process first tensor: layer norm + sigmoid
    sum_val1 = 0.0
    sum_sq1 = 0.0
    
    for i in range(hidden_size):
        idx = col_base + i
        x1 = tl.load(x1_ptr + idx)
        sum_val1 += x1
        sum_sq1 += x1 * x1
    
    mean1 = sum_val1 / hidden_size
    var1 = (sum_sq1 / hidden_size) - (mean1 * mean1)
    std1 = tl.sqrt(var1 + eps)
    
    # Process second tensor: layer norm
    sum_val2 = 0.0
    sum_sq2 = 0.0
    
    for i in range(hidden_size):
        idx = col_base + i
        x2 = tl.load(x2_ptr + idx)
        sum_val2 += x2
        sum_sq2 += x2 * x2
    
    mean2 = sum_val2 / hidden_size
    var2 = (sum_sq2 / hidden_size) - (mean2 * mean2)
    std2 = tl.sqrt(var2 + eps)
    
    # Compute layer norm outputs and sigmoid
    for i in range(hidden_size):
        idx = col_base + i
        
        # First tensor: layer norm + sigmoid
        x1 = tl.load(x1_ptr + idx)
        ln_out1 = (x1 - mean1) / std1 * ln_weight1 + ln_bias1
        sigmoid_out = 1.0 / (1.0 + tl.exp(-ln_out1))
        
        # Second tensor: layer norm
        x2 = tl.load(x2_ptr + idx)
        ln_out2 = (x2 - mean2) / std2 * ln_weight2 + ln_bias2
        
        # Store results
        tl.store(sigmoid_out_ptr + idx, sigmoid_out)
        tl.store(unsqueeze_out_ptr + idx, ln_out2)

@torch.fx.wrap
def fused_layer_norm_sigmoid_unsqueeze(in_x1, ln_weight1, ln_bias1, in_x2, ln_weight2, ln_bias2):
    # Get tensor shapes
    x1_elements = in_x1.numel()
    hidden_size = in_x1.shape[-1]
    
    # Determine grid size
    n_cols = tl.cdiv(x1_elements, hidden_size)
    num_programs = n_cols  # One program per column
    
    # Allocate output tensors
    sigmoid_out = torch.empty_like(in_x1)
    unsqueeze_out = torch.empty_like(in_x2)
    
    # Launch kernel
    layer_norm_sigmoid_unsqueeze_kernel[(num_programs,)](
        x1_ptr=in_x1,
        ln_weight1_ptr=ln_weight1,
        ln_bias1_ptr=ln_bias1,
        x2_ptr=in_x2,
        ln_weight2_ptr=ln_weight2,
        ln_bias2_ptr=ln_bias2,
        sigmoid_out_ptr=sigmoid_out,
        unsqueeze_out_ptr=unsqueeze_out,
        x1_elements=x1_elements,
        x2_elements=in_x2.numel(),
        hidden_size=hidden_size,
        UNSQUEEZE_DIM=-2,
        eps=1e-05
    )
    
    return sigmoid_out, unsqueeze_out

def replacement_func():
    return fused_layer_norm_sigmoid_unsqueeze