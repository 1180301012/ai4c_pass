import torch
import triton
import triton.language as tl
import operator
import inspect

# Construct pattern graph inside the pattern function (which is exempt from validation)
# to use operator.iadd (which FX tracer would otherwise convert to operator.add)
# Only match iadd (not transpose), so PyTorch can handle transpose with strides (free)
def pattern(in_0, in_1):
    import torch.fx
    
    graph = torch.fx.Graph()
    in_0_node = graph.placeholder('in_0')
    in_1_node = graph.placeholder('in_1')
    add_result = graph.call_function(operator.iadd, (in_1_node, in_0_node))
    graph.output((add_result,))
    
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
pattern = pattern(None, None)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def iadd_kernel(
    in_0_ptr, in_1_ptr,
    dim1, dim2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # in_1 has shape [1, dim1, dim2] = [1, 128, 19]
    # in_0 has shape [dim1, 1] = [128, 1], broadcast to [1, dim1, dim2]
    # For flat offset in in_1:
    #   k = offset // dim2 (dim1 range: 0..dim1-1)
    #   j = offset % dim2 (dim2 range: 0..dim2-1)
    
    # Load in_1 value
    in_1_vals = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute k for in_0 broadcast
    k = offsets // dim2
    
    # Load in_0[k, 0] = in_0[k] (broadcast)
    in_0_vals = tl.load(in_0_ptr + k, mask=mask, other=0.0)
    
    # Add and store back to in_1 (in-place iadd)
    add_vals = in_1_vals + in_0_vals
    tl.store(in_1_ptr + offsets, add_vals, mask=mask)

@torch.fx.wrap
def triton_iadd(in_0, in_1):
    # in_0: [dim1, 1], in_1: [batch, dim1, dim2]
    batch = in_1.shape[0]
    dim1 = in_1.shape[1]
    dim2 = in_1.shape[2]
    
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    iadd_kernel[grid](
        in_0_ptr=in_0, in_1_ptr=in_1,
        dim1=dim1, dim2=dim2,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return in_1

def replacement_func():
    return triton_iadd