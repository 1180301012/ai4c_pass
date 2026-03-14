import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern to match: mul -> batch_norm
    """
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.jit
def fused_mul_bn_kernel(
    x_ptr,
    scale_ptr,
    mean_ptr,
    var_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    total_elements,
    C, HW, CHW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < total_elements
    
    # Compute n and c indices
    n = offsets // CHW
    c = (offsets // HW) % C
    
    # Load x values - coalesced access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load scale using gather (broadcast from [N, C, 1, 1])
    scale_idx = n * C + c
    scale = tl.load(scale_ptr + scale_idx, mask=mask, other=0.0)
    
    # Load BN parameters using gather
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    gamma = tl.load(gamma_ptr + c, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + c, mask=mask, other=0.0)
    
    # Fused computation
    y = x * scale
    inv_std = tl.rsqrt(var + eps)
    y = (y - mean) * inv_std * gamma + beta
    
    tl.store(out_ptr + offsets, y, mask=mask)


@torch.fx.wrap
def fused_mul_bn(in_0, in_1, in_2, in_3, in_4, in_5):
    N, C, H, W = in_5.shape
    HW = H * W
    CHW = C * HW
    total_elements = N * CHW
    
    device = in_5.device
    mean = in_0.to(device).contiguous()
    var = in_1.to(device).contiguous()
    gamma = in_3.to(device).contiguous()
    beta = in_2.to(device).contiguous()
    x = in_5.contiguous()
    scale = in_4.contiguous()
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 4096
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_mul_bn_kernel[(num_programs,)](
        x_ptr=x,
        scale_ptr=scale,
        mean_ptr=mean,
        var_ptr=var,
        gamma_ptr=gamma,
        beta_ptr=beta,
        out_ptr=out,
        total_elements=total_elements,
        C=C, HW=HW, CHW=CHW,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return fused_mul_bn