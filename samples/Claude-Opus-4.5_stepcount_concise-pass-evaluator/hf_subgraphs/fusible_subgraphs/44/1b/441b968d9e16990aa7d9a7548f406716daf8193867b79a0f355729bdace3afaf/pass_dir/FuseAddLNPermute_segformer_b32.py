import torch
import triton
import triton.language as tl

# Pattern for add + layernorm + reshape + permute + contiguous (batch=32)
def pattern(bias, weight, x, y):
    tmp_2 = y + x
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (512,), weight, bias, 1e-05)
    tmp_4 = tmp_3.reshape(32, 16, 16, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(bias, weight, x, y):
    return (bias, weight, x, y)

@triton.jit
def fused_add_ln_permute_kernel_b32(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    B, N,  # batch (32), sequence length (256)
    C: tl.constexpr,  # channels (512)
    H: tl.constexpr,  # height (16)
    W: tl.constexpr,  # width (16)
    eps: tl.constexpr,
):
    # Each program processes one row (one spatial position across all batches)
    pid = tl.program_id(0)
    
    # Compute batch and sequence indices
    batch_idx = pid // (H * W)
    hw_idx = pid % (H * W)
    
    # Input offset: input[batch, seq, channel]
    in_offset = batch_idx * (H * W * C) + hw_idx * C
    
    # Channel offsets
    offs_c = tl.arange(0, C)
    
    # Load input data
    x = tl.load(x_ptr + in_offset + offs_c)
    y = tl.load(y_ptr + in_offset + offs_c)
    
    # Add
    sum_val = x + y
    
    # LayerNorm
    sum_val_f32 = sum_val.to(tl.float32)
    mean = tl.sum(sum_val_f32, axis=0) / C
    diff = sum_val_f32 - mean
    var = tl.sum(diff * diff, axis=0) / C
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias
    w = tl.load(weight_ptr + offs_c)
    b = tl.load(bias_ptr + offs_c)
    
    # Apply affine transformation
    normalized = diff * inv_std * w.to(tl.float32) + b.to(tl.float32)
    
    # Compute output position in NCHW layout
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
    # Output offset: out[batch, c, h, w]
    out_base = batch_idx * (C * H * W) + h_idx * W + w_idx
    out_stride_c = H * W
    
    tl.store(out_ptr + out_base + offs_c * out_stride_c, normalized.to(sum_val.dtype))

@torch.fx.wrap
def fused_add_ln_permute_b32(bias, weight, x, y):
    # bias: [512], weight: [512]
    # x, y: [32, 256, 512]
    # output: [32, 512, 16, 16]
    
    B = x.shape[0]  # 32
    N = x.shape[1]  # 256
    C = x.shape[2]  # 512
    H = 16
    W = 16
    
    # Ensure inputs are contiguous and on same device
    x_c = x.contiguous()
    y_c = y.contiguous()
    weight_c = weight.to(x.device).contiguous()
    bias_c = bias.to(x.device).contiguous()
    
    # Allocate output in NCHW format
    out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    
    # Launch kernel - one block per spatial position per batch
    grid = (B * N,)
    fused_add_ln_permute_kernel_b32[grid](
        y_c, x_c, weight_c, bias_c, out,
        B, N, C=C, H=H, W=W, eps=1e-05,
        num_warps=8,
    )
    
    return out

def replacement_func():
    return fused_add_ln_permute_b32