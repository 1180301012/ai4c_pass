import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern for fusing sum(dim=3, keepdim=True) + division operations
    This matches: tmp_5 = in_3.sum(dim = 3, keepdim = True); tmp_6 = in_3 / tmp_5
    """
    tmp_5 = x.sum(dim=3, keepdim=True)
    out = x / tmp_5
    return out

def replacement_args(x):
    """Extract the input tensor for the fused normalization operation"""
    return (x,)

@triton.jit
def compute_sum_along_dim3_kernel(
    x_ptr,
    sum_ptr,
    N, C, H,
    W: tl.constexpr,
):
    """
    Kernel to compute sum along dim=3 (last dimension) with keepdim=True
    For input shape [N,C,H,W], produces output shape [N,C,H,1]
    """
    # Each program handles one element in the N,C,H space
    pid = tl.program_id(0)
    
    # Calculate N, C, H indices from flat pid
    # Total elements in N,C,H space: N * C * H
    if pid >= N * C * H:
        return
        
    ph = pid % H
    pid = pid // H
    pc = pid % C  
    pn = pid // C
    
    # Load entire W dimension for this N,C,H position
    base_offset = pn * (C * H * W) + pc * (H * W) + ph * W
    offsets = base_offset + tl.arange(0, W)
    
    # Load all W elements and compute sum
    x_data = tl.load(x_ptr + offsets)
    sum_val = tl.sum(x_data)
    
    # Store sum to output tensor at position [N,C,H,1]  
    sum_offset = pn * (C * H) + pc * H + ph
    tl.store(sum_ptr + sum_offset, sum_val)

@torch.fx.wrap  
def fused_normalization(x):
    """
    Compute fused normalization: x / x.sum(dim=3, keepdim=True)
    """
    N, C, H, W = x.shape
    
    # Output tensor for the sum (shape N,C,H,1)
    sum_out = torch.empty(N, C, H, 1, dtype=x.dtype, device=x.device, requires_grad=False)
    
    # Step 1: Compute sums along dim=3 using Triton kernel
    # Each program handles one element in the N,C,H space
    total_elements = N * C * H
    compute_sum_along_dim3_kernel[(total_elements,)](
        x_ptr=x,
        sum_ptr=sum_out.view(N * C * H),  # Flattened to 1D for easier addressing
        N=N, C=C, H=H, W=W,
    )
    
    # Step 2: Complete the normalization using PyTorch for simplicity
    # Expand the sum to match original shape for broadcasting
    sum_expanded = sum_out.expand(-1, -1, -1, W)
    result = x / sum_expanded
    
    return result

def replacement_func():
    """Return the fused normalization function"""
    return fused_normalization