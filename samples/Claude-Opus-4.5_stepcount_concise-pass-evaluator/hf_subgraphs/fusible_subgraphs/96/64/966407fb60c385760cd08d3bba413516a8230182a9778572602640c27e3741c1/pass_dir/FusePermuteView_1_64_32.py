import torch
import triton
import triton.language as tl

# Pattern to match: permute -> contiguous -> view for graph 0
# View shape: [1, 4, 64, 8] -> permute -> [1, 64, 4, 8] -> view -> [1, 64, 32]
def pattern(input_tensor):
    permuted = input_tensor.permute(0, 2, 1, 3)
    contiguous = permuted.contiguous()
    viewed = contiguous.view(1, 64, 32)
    return (viewed,)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def fused_permute_view_kernel_0(
    input_ptr,
    out_ptr,
    N, C, H, W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: permute(0,2,1,3) + view to flatten last 2 dims
    Input: [N, C, H, W], Output: [N, H, C*W]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Output is [N, H, C*W]
    CW = C * W
    cw = offsets % CW
    temp = offsets // CW
    h = temp % H
    n = temp // H
    
    c = cw // W
    w = cw % W
    
    # Input index in [N, C, H, W] layout
    in_idx = n * (C * H * W) + c * (H * W) + h * W + w
    
    val = tl.load(input_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def fused_permute_view_0(input_tensor):
    N, C, H, W = input_tensor.shape
    n_elements = N * C * H * W
    
    out = torch.empty(N, H, C * W, device=input_tensor.device, dtype=input_tensor.dtype)
    
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_permute_view_kernel_0[grid](
        input_ptr=input_tensor,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_permute_view_0