import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_bn_silu_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    total_elements,
    C, HW, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused batch_norm + SiLU kernel.
    
    Computes: silu(batch_norm(input)) in a single pass.
    batch_norm in inference mode: output = weight * (input - mean) / sqrt(var + eps) + bias
    SiLU: silu(x) = x * sigmoid(x)
    
    All computation done in float32 for accuracy, then cast back to original dtype on store.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input and cast to float32
    x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    
    # Compute channel index: for [1, C, H, W] layout, channel = offset // (H*W)
    c = offsets // HW
    
    # Load BN parameters per channel and cast to float32
    m = tl.load(mean_ptr + c, mask=mask).to(tl.float32)
    v = tl.load(var_ptr + c, mask=mask).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask).to(tl.float32)
    
    # Compute batch_norm: scale * x + offset
    # scale = weight / sqrt(var + eps)
    # offset = bias - mean * scale
    scale = w / tl.sqrt(v + eps)
    offset_val = b - m * scale
    bn_out = scale * x + offset_val
    
    # Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    silu_out = bn_out * tl.sigmoid(bn_out)
    
    # Store (Triton auto-casts float32 to output tensor dtype)
    tl.store(output_ptr + offsets, silu_out, mask=mask)


@torch.fx.wrap
def fused_bn_silu_dispatch(mean, var, bias, weight, input_tensor, route):
    """Dispatch wrapper for fused BN + SiLU computation.
    
    Handles different reshape configurations based on route string.
    Route strings:
      - "bn_silu_256_16_16": reshape to (1, 256, 16, 16)
      - "bn_silu_512_8_8": reshape to (1, 512, 8, 8)
    """
    if route == "bn_silu_256_16_16":
        x = input_tensor.reshape(1, 256, 16, 16)
        C = 256
        HW = 256  # 16 * 16
    elif route == "bn_silu_512_8_8":
        x = input_tensor.reshape(1, 512, 8, 8)
        C = 512
        HW = 64   # 8 * 8
    else:
        raise ValueError(f"Unknown route: {route}")
    
    # Ensure all tensors are contiguous and on the same device
    x = x.contiguous()
    device = x.device
    
    mean_gpu = mean.to(device=device).contiguous()
    var_gpu = var.to(device=device).contiguous()
    weight_gpu = weight.to(device=device).contiguous()
    bias_gpu = bias.to(device=device).contiguous()
    
    total_elements = x.numel()
    eps = 1e-05
    
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    fused_bn_silu_kernel[grid](
        input_ptr=x, output_ptr=output,
        mean_ptr=mean_gpu, var_ptr=var_gpu, weight_ptr=weight_gpu, bias_ptr=bias_gpu,
        total_elements=total_elements,
        C=C, HW=HW, eps=eps,
    )
    
    return output