import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_op_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    H, W, F,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    total_elements = F * H * W
    for idx in range(block_start, min(block_end, total_elements)):
        f = idx // (H * W)
        h = (idx % (H * W)) // W
        w = idx % W
        
        # Get softmax weights from in_1 (2x256)
        in_1_0 = tl.load(in_1_ptr + 0 * F + f)
        in_1_1 = tl.load(in_1_ptr + 1 * F + f)
        
        # Stable softmax for dim=1 (size 2)
        max_val = in_1_0 if in_1_0 > in_1_1 else in_1_1
        x0 = in_1_0 - max_val
        x1 = in_1_1 - max_val
        exp_x0 = tl.exp(x0.to(tl.float32)).to(x0.dtype)
        exp_x1 = tl.exp(x1.to(tl.float32)).to(x1.dtype)
        sum_exp = exp_x0 + exp_x1
        weight0 = exp_x0 / sum_exp
        weight1 = exp_x1 / sum_exp
        
        # Get values from in_0 (2x256xHxW)
        in_0_0 = tl.load(in_0_ptr + 0 * (F * H * W) + f * (H * W) + h * W + w)
        in_0_1 = tl.load(in_0_ptr + 1 * (F * H * W) + f * (H * W) + h * W + w)
        
        # Compute weighted sum
        out_val = weight0 * in_0_0 + weight1 * in_0_1
        tl.store(out_ptr + idx, out_val)

@torch.fx.wrap
def fused_op(in_0, in_1):
    # Flatten tensors for kernel processing

    
    # Extract dimensions
    B, C, F, H, W = in_0.shape
    assert B == 1 and C == 2, "Invalid shape for in_0"
    
    # Allocate output [F, H, W]
    out = torch.empty((F, H, W), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    BLOCK_SIZE = 128
    num_blocks = (F * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_op_kernel[(num_blocks,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        H=H, W=W, F=F,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.unsqueeze(0)  # [1, 256, H, W]

def replacement_func():
    return fused_op