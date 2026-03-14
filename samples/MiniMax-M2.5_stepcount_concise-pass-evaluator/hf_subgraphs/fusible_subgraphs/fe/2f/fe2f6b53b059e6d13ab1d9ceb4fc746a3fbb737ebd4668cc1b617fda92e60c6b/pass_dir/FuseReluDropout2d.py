import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
    ],
    key=['N'],
)
@triton.jit
def relu_dropout2d_kernel(
    input_ptr,
    output_dropout_ptr,
    output_relu_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Custom kernel for fused ReLU + Dropout2d.
    Since Dropout2d with training=False is a no-op, we compute ReLU and return it twice.
    """
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N  # Mask to ensure we don't go out of bounds

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute ReLU: max(0, x)
    relu_out = tl.where(x > 0, x, 0.0)

    # Store both outputs (dropout output is same as relu output when training=False)
    tl.store(output_dropout_ptr + offsets, relu_out, mask=mask)
    tl.store(output_relu_ptr + offsets, relu_out, mask=mask)


def pattern(in_0):
    """
    Match the pattern: ReLU(inplace=True) -> Dropout2d(training=False)
    The dropout2d with training=False is a no-op, so we can fuse this.
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)


def replacement_args(in_0):
    return (in_0,)


@torch.fx.wrap
def relu_dropout2d_wrapper(in_0):
    """
    Wrapper function that launches the Triton kernel.
    Since Dropout2d with training=False returns the input unchanged,
    we compute ReLU once and return it for both outputs.
    """
    N = in_0.numel()
    
    # Choose BLOCK_SIZE based on tensor size
    if N <= 4096:
        BLOCK_SIZE = 1024
    elif N <= 16384:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create output tensors
    output_dropout = torch.empty_like(in_0)
    output_relu = torch.empty_like(in_0)

    # Flatten for kernel
    in_0_flat = in_0.view(-1)
    output_dropout_flat = output_dropout.view(-1)
    output_relu_flat = output_relu.view(-1)

    # Launch kernel
    relu_dropout2d_kernel[(num_programs,)](
        input_ptr=in_0_flat,
        output_dropout_ptr=output_dropout_flat,
        output_relu_ptr=output_relu_flat,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Reshape back to original shape
    output_dropout = output_dropout.view(in_0.shape)
    output_relu = output_relu.view(in_0.shape)

    return (output_dropout, output_relu)


def replacement_func():
    return relu_dropout2d_wrapper