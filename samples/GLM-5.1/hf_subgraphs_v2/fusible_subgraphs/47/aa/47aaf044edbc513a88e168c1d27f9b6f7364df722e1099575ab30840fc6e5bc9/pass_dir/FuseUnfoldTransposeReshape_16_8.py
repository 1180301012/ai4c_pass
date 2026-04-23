import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5


def replacement_args(in_0):
    return (in_0, "tiny")


@triton.jit
def fused_unfold_reshape_kernel(
    input_ptr,
    output_ptr,
    C,
    W,
    G,
    K: tl.constexpr,
    PAD: tl.constexpr,
    stride_c,
    stride_w,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Decompose flat output index into (i, j, k)
    # output shape: [N, G, K] where N = C * W // G
    # flat index = i * G * K + j * K + k
    k = offsets % K
    j = (offsets // K) % G
    i = offsets // (K * G)

    # Map to input coordinates
    # input shape: [1, C, W]
    c = (i * G + j) % C
    w = (i * G + j) // C + k - PAD

    # Check bounds (zero-padding)
    valid = (w >= 0) & (w < W)
    input_offset = c * stride_c + w * stride_w

    # Load input value (0 if out of bounds due to padding)
    val = tl.load(input_ptr + input_offset, mask=valid & mask, other=0.0)

    # Store output
    tl.store(output_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def _fused_unfold_impl(input_tensor, G):
    C = input_tensor.shape[1]
    W = input_tensor.shape[2]
    K = 9
    PAD = 4
    N = C * W // G
    stride_c = input_tensor.stride(1)
    stride_w = input_tensor.stride(2)

    output = torch.empty((N, G, K), dtype=input_tensor.dtype, device=input_tensor.device)
    total_elements = N * G * K
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_unfold_reshape_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        C=C,
        W=W,
        G=G,
        K=K,
        PAD=PAD,
        stride_c=stride_c,
        stride_w=stride_w,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


@torch.fx.wrap
def _fused_unfold_tiny(input_tensor):
    return _fused_unfold_impl(input_tensor, 8)


@torch.fx.wrap
def _fused_unfold_base(input_tensor):
    return _fused_unfold_impl(input_tensor, 64)


@torch.fx.wrap
def dispatch_wrapper(input_tensor, route):
    if route == "tiny":
        return _fused_unfold_tiny(input_tensor)
    elif route == "base":
        return _fused_unfold_base(input_tensor)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    return dispatch_wrapper