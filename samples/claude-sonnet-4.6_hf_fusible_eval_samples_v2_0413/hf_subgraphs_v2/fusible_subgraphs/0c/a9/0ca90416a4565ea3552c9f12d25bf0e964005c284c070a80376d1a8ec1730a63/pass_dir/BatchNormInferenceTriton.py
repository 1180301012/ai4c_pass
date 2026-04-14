import torch
import triton
import triton.language as tl


@triton.jit
def _bn_inference_kernel(
    input_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    C,
    HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Batch-norm inference kernel.
    2-D grid: (N*C, ceil(HW / BLOCK_SIZE))

    Two BLOCK_SIZE variants compiled ahead of time during warmup:
      BS=256, num_warps=4  →  2 elements/thread, 16 concurrent blocks/SM
      BS=512, num_warps=2  →  8 elements/thread, 32 concurrent blocks/SM (max A30)

    Selecting BS=512 when HW%512==0 enables 128-bit (v4.u32) vectorised
    loads for bfloat16/float16 – each thread loads 8×2=16 bytes in ONE
    memory instruction instead of 4×2=8 bytes (BS=256).  The 32-block/SM
    occupancy doubles latency-hiding opportunities over the 16-block case.

    2-D grid: program_id(0) = nc (uses %ctaid.x, lowest-latency register),
    program_id(1) = hw_block (uses %ctaid.y).  No %ctaid.z involved.
    Consecutive nc addresses stride HW*sizeof(T) ≈ 4608 B for HW=2304 bf16,
    inside the GPU's stride-prefetcher window.
    """
    nc       = tl.program_id(0)
    hw_block = tl.program_id(1)

    c = nc % C

    mean_c   = tl.load(mean_ptr   + c).to(tl.float32)
    var_c    = tl.load(var_ptr    + c).to(tl.float32)
    weight_c = tl.load(weight_ptr + c).to(tl.float32)
    bias_c   = tl.load(bias_ptr   + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var_c + eps)
    scale   = weight_c * inv_std
    shift   = bias_c   - mean_c * scale

    hw_offsets = hw_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask       = hw_offsets < HW
    offsets    = nc * HW + hw_offsets

    x     = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    y_f32 = x_f32 * scale + shift
    tl.store(output_ptr + offsets, y_f32.to(x.dtype), mask=mask)


@torch.fx.wrap
def _triton_bn_forward(in_0, in_1, in_2, in_3, in_4):
    """
    Drop-in replacement for torch.nn.functional.batch_norm (inference only).

    Argument order follows the pattern (matches model.py calls):
        in_0 : running_mean   [C]
        in_1 : running_var    [C]
        in_2 : bias           [C]
        in_3 : weight         [C]
        in_4 : input          [N, C, H, W]
    """
    input_tensor = in_4
    dev = input_tensor.device

    running_mean = torch.as_tensor(in_0, device=dev)
    running_var  = torch.as_tensor(in_1, device=dev)
    weight       = torch.as_tensor(in_3, device=dev)
    bias         = torch.as_tensor(in_2, device=dev)

    N, C, H, W = input_tensor.shape
    HW  = H * W
    NC  = N * C
    eps = 0.001

    output = torch.empty_like(input_tensor)

    # Two pre-compiled variants; each triggered during a different warmup phase,
    # so no mid-timing JIT compilation.
    #
    # BS=512, nw=2: HW % 512 == 0  → 8 elems/thread, 128-bit loads, 32 blocks/SM
    # BS=256, nw=4: all other HW   → 2 elems/thread,  32-bit loads, 16 blocks/SM
    if HW % 512 == 0:
        BLOCK_SIZE = 512
        NUM_WARPS  = 2   # 64 threads → 8 bf16/fp16 per thread → 128-bit load
    else:
        BLOCK_SIZE = 256
        NUM_WARPS  = 4   # 128 threads → 2 bf16/fp16 per thread → 32-bit load

    grid = (NC, triton.cdiv(HW, BLOCK_SIZE))

    _bn_inference_kernel[grid](
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        C,
        HW,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )

    return output


# ---------------------------------------------------------------------------
# Pattern / replacement interface expected by the AI4C evaluation framework
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Matches:
        torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2,
                                        False, 0.1, 0.001)
    which appears in every target graph as the inference-mode BN call.
    """
    return torch.nn.functional.batch_norm(
        in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001
    )


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return _triton_bn_forward