import torch
import triton
import triton.language as tl

# Pattern for face-parsing batch=32
def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_2 = in_4 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (512,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3.reshape(32, 16, 16, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = in_2.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return (tmp_6, tmp_8)

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def fused_add_ln_permute_kernel_fp_b32(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    B, N, C,
    H: tl.constexpr, W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Input offset for this row
    in_offset = pid * C
    
    # Load channel data
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C
    
    x = tl.load(x_ptr + in_offset + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + in_offset + offs, mask=mask, other=0.0)
    
    # Add
    sum_val = x + y
    
    # LayerNorm: compute mean
    mean = tl.sum(sum_val, axis=0) / C
    
    # Compute variance
    diff = sum_val - mean
    var = tl.sum(diff * diff, axis=0) / C
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    
    # Apply affine transformation
    normalized = diff * inv_std * weight + bias
    
    # Output in NCHW format
    batch_idx = pid // (H * W)
    hw_idx = pid % (H * W)
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
    out_base = batch_idx * C * H * W + h_idx * W + w_idx
    out_stride_c = H * W
    
    tl.store(out_ptr + out_base + offs * out_stride_c, normalized, mask=mask)

@torch.fx.wrap
def fused_op_faceparse_b32(in_0, in_1, in_2, in_3, in_4):
    # in_0: bias [512]
    # in_1: weight [512]
    # in_2: [32, 64, 128, 128] - for flatten
    # in_3: [32, 256, 512] - for add
    # in_4: [32, 256, 512] - for add
    
    B = in_3.shape[0]
    N = in_3.shape[1]  # 256
    C = in_3.shape[2]  # 512
    H = 16
    W = 16
    
    # Ensure inputs are contiguous and on same device
    x = in_4.contiguous()
    y = in_3.contiguous()
    weight = in_1.to(in_3.device).contiguous()
    bias = in_0.to(in_3.device).contiguous()
    
    # Allocate output for fused operation (NCHW format)
    out1 = torch.empty((B, C, H, W), device=in_3.device, dtype=in_3.dtype)
    
    # Launch kernel
    grid = (B * N,)
    fused_add_ln_permute_kernel_fp_b32[grid](
        x, y, weight, bias, out1,
        B, N, C,
        H, W,
        eps=1e-05,
    )
    
    # Second output: flatten + transpose (using in_2)
    out2 = in_2.flatten(2).transpose(1, 2)
    
    return (out1, out2)

def replacement_func():
    return fused_op_faceparse_b32