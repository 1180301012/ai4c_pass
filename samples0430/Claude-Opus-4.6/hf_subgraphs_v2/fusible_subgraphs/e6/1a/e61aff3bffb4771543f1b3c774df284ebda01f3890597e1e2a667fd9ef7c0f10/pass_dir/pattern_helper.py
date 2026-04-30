import torch
import torch.fx
import torch.nn.functional
import inspect


def _build_pattern_graph():
    g = torch.fx.Graph()
    in_0 = g.placeholder('in_0')
    in_1 = g.placeholder('in_1')
    in_2 = g.placeholder('in_2')

    # conv2d: call_function with all positional args (matching dynamo's representation)
    conv = g.call_function(
        torch.conv2d,
        args=(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    )

    # view: call_method
    view = g.call_method('view', args=(conv, 4, 1, 192))

    # softmax: call_function with EXACT args/kwargs as dynamo produces
    softmax = g.call_function(
        torch.nn.functional.softmax,
        args=(view, 2),
        kwargs={'_stacklevel': 5}
    )

    # unsqueeze: call_method
    unsqueeze = g.call_method('unsqueeze', args=(softmax, -1))

    g.output((unsqueeze,))
    return g


class _DummyModule(torch.nn.Module):
    pass


pattern_gm = torch.fx.GraphModule(_DummyModule(), _build_pattern_graph())
pattern_gm.__signature__ = inspect.Signature(
    parameters=[
        inspect.Parameter('in_0', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_1', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('in_2', inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
)