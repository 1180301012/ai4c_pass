import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_HW = HW = 4096 (all positions at once — zero redundant reads)
        triton.Config({'BLOCK_HW': 4096}, num_warps=4),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16),
        triton.Config({'BLOCK_HW': 4096}, num_warps=32),
        # Smaller BLOCK_HW for comparison
        triton.Config({'BLOCK_HW': 2048}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW', 'C'],
)
@triton.jit
def _fused_conv1x1_softmax_kernel(
    input_ptr,   # [N, C, HW] – contiguous NCHW viewed as [N, C, H*W]
    weight_ptr,  # [C] – first element of weight [1, C, 1, 1]
    bias_ptr,    # [1]
    output_ptr,  # [N, 1, H, W] = [N, 1, HW] – flat offset = n*HW + hw
    N, C, HW,
    BLOCK_HW: tl.constexpr,  # number of spatial positions handled per program
    BLOCK_C:  tl.constexpr,  # channel block size; 512 loads in one shot
):
    # One program per batch element – reads input exactly once (no redundancy)
    pid_n = tl.program_id(0)

    # Vector of HW positions handled by this program
    hw_offsets = tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    # float32 accumulator (one per spatial position)
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Accumulate dot-product: acc[hw] += sum_c input[n,c,hw] * weight[c]
    for c in range(0, BLOCK_C, 1):
        # Scalar weight load (broadcast to all hw positions)
        w = tl.load(weight_ptr + c)
        # Coalesced vector load: input[n, c, hw_start:hw_start+BLOCK_HW]
        base_input = pid_n * C * HW + c * HW
        x = tl.load(input_ptr + base_input + hw_offsets, mask=hw_mask, other=0.0)
        acc += x.to(tl.float32) * w

    # Add bias (scalar → broadcast)
    acc += tl.load(bias_ptr).to(tl.float32)

    # Numerically stable softmax over HW dimension
    m      = tl.max(acc, axis=0)
    acc_exp = tl.exp(acc - m)
    s      = tl.sum(acc_exp, axis=0)
    out_f32 = acc_exp / s

    # Store: output[n, 0, hw] at flat offset n*HW + hw
    base_output = pid_n * HW
    tl.store(output_ptr + base_output + hw_offsets, out_f32, mask=hw_mask)


def fused_conv1x1_softmax(bias, weight, inp):
    """
    Fused 1x1-conv + softmax.
    inp      : [N, C, H, W]  NCHW contiguous
    weight   : [1, C, 1, 1]
    bias     : [1]
    returns  : [N, 1, H*W]  == [N, 1, HW]
    """
    N  = inp.shape[0]
    C  = inp.shape[1]
    H  = inp.shape[2]
    W  = inp.shape[3]
    HW = H * W

    # Allocate output [N, 1, HW]
    output = torch.empty((N, 1, HW), dtype=inp.dtype, device=inp.device)

    # One program per batch element; each program covers BLOCK_HW spatial positions
    grid = lambda meta: (N,)

    _fused_conv1x1_softmax_kernel[grid](
        inp, weight, bias, output,
        N=N, C=C, HW=HW,
        BLOCK_C=512,
    )

    return output