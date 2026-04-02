import torch
import triton
import triton.language as tl


# Pattern: relu(x) + adaptive_avg_pool2d + flatten
# Using inplace=False to get a proper (non-dead) relu node in the FX graph.
# The SubgraphMatcher may ignore the inplace kwarg difference vs the target's inplace=True.
def pattern(x):
    tmp_5 = torch.nn.functional.relu(x, inplace=False)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7


def replacement_args(x):
    # x is the INPUT to relu (pre-activation tensor)
    return (x,)


@triton.jit
def fused_relu_avgpool_kernel(
    x_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    One program per channel.
    Computes: out[c] = mean(relu(x[c, 0..HW-1]))
    Fuses relu + global average pooling into a single pass.
    """
    c = tl.program_id(0)
    base = c * HW
    offsets = tl.arange(0, BLOCK_HW)
    mask = offsets < HW
    # Load and apply relu
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    val = tl.maximum(x, 0.0)
    # Global average pool (masked zeros don't affect sum since relu(0)=0)
    avg = tl.sum(val, axis=0) / HW
    tl.store(out_ptr + c, avg.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_relu_avgpool_wrapper(x):
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3]
    out = torch.empty((1, C), dtype=x.dtype, device=x.device)
    fused_relu_avgpool_kernel[(C,)](
        x, out, C, HW, BLOCK_HW=256, num_warps=8
    )
    return out


def replacement_func():
    return fused_relu_avgpool_wrapper