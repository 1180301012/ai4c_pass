import inspect
import torch
import triton
import triton.language as tl


# Build the pattern as a concrete FX graph because symbolic tracing a zero-input
# Python function would constant-fold torch.arange instead of recording a node.
_pattern_graph = torch.fx.Graph()
_arange_node = _pattern_graph.call_function(
    torch.arange,
    args=(1,),
    kwargs={"device": torch.device(type='cuda', index=0)},
)
_pattern_graph.output(_arange_node)
pattern = torch.fx.GraphModule(torch.nn.Module(), _pattern_graph)
pattern.__signature__ = inspect.Signature(parameters=[])



def replacement_args():
    return tuple()


@triton.jit
def _dummy_kernel(out_ptr):
    tl.store(out_ptr, 0)


@torch.fx.wrap
def _replace_arange1_cuda_with_const_tensor():
    # arange(1, device='cuda') == tensor([0], device='cuda', dtype=int64)
    return torch.zeros((1,), device='cuda', dtype=torch.int64)



def replacement_func():
    return _replace_arange1_cuda_with_const_tensor