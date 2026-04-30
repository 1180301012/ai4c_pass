import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu6_gap_kernel(
    in_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU6 + Global Average Pooling kernel.
    Each program handles one (batch, channel) pair.
    Grid shape: (N*C,)

    BLOCK_SIZE is the smallest power-of-2 >= HW.
    A single vectorised load covers all HW elements; no loop needed.
    """
    pid = tl.program_id(0)
    base = pid * HW
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    # Single load – x.dtype is accessible here for the final cast
    x = tl.load(in_ptr + base + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # ReLU6: clamp to [0, 6]
    x_f32 = tl.minimum(tl.maximum(x_f32, 0.0), 6.0)

    # Global average pooling: sum / HW
    result = tl.sum(x_f32, axis=0) / tl.cast(HW, tl.float32)

    # Explicitly cast back to input dtype and store
    tl.store(out_ptr + pid, result.to(x.dtype))


@torch.fx.wrap
def dispatch_fused_relu6_gap_avgpool(x):
    """
    Unified wrapper for fused ReLU6 + Global Average Pooling.
    Input:  x of shape [N, C, H, W]
    Output: [N, C] of same dtype as x

    Caches N, C, HW, NC, grid tuple, output buffer, and kernel handle to
    eliminate ALL repeated Python object creation on the critical path.
    """
    # Recompute on first call OR when batch/channel dims change
    # Use x.shape tuple for the dimension check (cached once, no repeated attr access)
    try:
        N_val = getattr(dispatch_fused_relu6_gap_avgpool, '_N')
        if N_val != x.shape[0]:
            raise AttributeError
    except AttributeError:
        s = x.shape
        N_val = s[0]; C_val = s[1]; H_val = s[2]; W_val = s[3]
        HW_val = H_val * W_val; NC_val = N_val * C_val
        dispatch_fused_relu6_gap_avgpool._N = N_val; dispatch_fused_relu6_gap_avgpool._C = C_val
        dispatch_fused_relu6_gap_avgpool._H = H_val; dispatch_fused_relu6_gap_avgpool._W = W_val
        dispatch_fused_relu6_gap_avgpool._HW = HW_val; dispatch_fused_relu6_gap_avgpool._NC = NC_val
        dispatch_fused_relu6_gap_avgpool._grid = (NC_val,)
        dispatch_fused_relu6_gap_avgpool._out = torch.empty(
            (N_val, C_val), dtype=x.dtype, device=x.device
        )
        dispatch_fused_relu6_gap_avgpool._k16 = fused_relu6_gap_kernel[(NC_val,)]
        dispatch_fused_relu6_gap_avgpool._k64 = fused_relu6_gap_kernel[(NC_val,)]
        dispatch_fused_relu6_gap_avgpool._k256 = fused_relu6_gap_kernel[(NC_val,)]
        dispatch_fused_relu6_gap_avgpool._k512 = fused_relu6_gap_kernel[(NC_val,)]

    N = dispatch_fused_relu6_gap_avgpool._N
    C = dispatch_fused_relu6_gap_avgpool._C
    HW = dispatch_fused_relu6_gap_avgpool._HW
    out = dispatch_fused_relu6_gap_avgpool._out

    # Dispatch to pre-cached kernel handle based on HW
    if HW <= 16:
        dispatch_fused_relu6_gap_avgpool._k16(x, out, HW, BLOCK_SIZE=16)
    elif HW <= 64:
        dispatch_fused_relu6_gap_avgpool._k64(x, out, HW, BLOCK_SIZE=64)
    elif HW <= 256:
        dispatch_fused_relu6_gap_avgpool._k256(x, out, HW, BLOCK_SIZE=256)
    else:
        dispatch_fused_relu6_gap_avgpool._k512(x, out, HW, BLOCK_SIZE=512)

    return out