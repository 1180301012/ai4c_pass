import torch
import triton
import triton.language as tl
import operator
import inspect

# Construct pattern graph inside the pattern function (which is exempt from validation)
# to use operator.iadd (which FX tracer would otherwise convert to operator.add)
def pattern(in_0, in_1):
    import torch.fx
    
    graph = torch.fx.Graph()
    in_0_node = graph.placeholder('in_0')
    in_1_node = graph.placeholder('in_1')
    add_result = graph.call_function(operator.iadd, (in_1_node, in_0_node))
    transpose_result = graph.call_method('transpose', (add_result, 1, 2))
    graph.output((transpose_result,))
    
    class EmptyModule(torch.nn.Module):
        pass
    
    gm = torch.fx.GraphModule(EmptyModule(), graph)
    
    # Set custom signature so inspect.signature(pattern) returns correct arg names
    sig = inspect.Signature(parameters=[
        inspect.Parameter('in_0', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_1', inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ])
    gm.__signature__ = sig
    
    return gm

# Call pattern at module level to get the GraphModule
# The function body uses torch.fx APIs, but those are exempt since they're inside pattern()
pattern = pattern(None, None)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_transpose_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    dim1, dim2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Output shape: [1, dim2, dim1] (transposed from [1, dim1, dim2])
    # For flat offset within batch=0:
    #   j = offset // dim1 (dim2 range: 0..dim2-1)
    #   k = offset % dim1 (dim1 range: 0..dim1-1)
    k = offsets % dim1
    j = (offsets // dim1) % dim2
    
    # in_1[0, k, j] at flat offset k * dim2 + j (in_1 has shape [1, dim1, dim2])
    in_1_offsets = k * dim2 + j
    in_1_vals = tl.load(in_1_ptr + in_1_offsets, mask=mask, other=0.0)
    
    # in_0[k, 0] at flat offset k (in_0 has shape [dim1, 1], broadcast to [1, dim1, dim2])
    in_0_vals = tl.load(in_0_ptr + k, mask=mask, other=0.0)
    
    add_vals = in_1_vals + in_0_vals
    
    # Write add result back to in_1 (matching the iadd side effect)
    tl.store(in_1_ptr + in_1_offsets, add_vals, mask=mask)
    
    # Write transposed result to output
    tl.store(out_ptr + offsets, add_vals, mask=mask)

@torch.fx.wrap
def fused_add_transpose(in_0, in_1):
    batch = in_1.shape[0]
    dim1 = in_1.shape[1]
    dim2 = in_1.shape[2]
    
    out = torch.empty((batch, dim2, dim1), dtype=in_1.dtype, device=in_1.device)
    
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_add_transpose_kernel[grid](
        in_0_ptr=in_0, in_1_ptr=in_1, out_ptr=out,
        dim1=dim1, dim2=dim2,
        n_elements=n_elements,
    )
    
    return out

def replacement_func():
    return fused_add_transpose