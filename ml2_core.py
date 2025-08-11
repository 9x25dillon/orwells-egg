import torch
import torch.nn as nn
import torch.nn.functional as F


class CompoundNode(nn.Module):
    """
    Combines multiple simple activations; mixing weights are softmax-normalized
    to enforce non-negative, sum-to-one interpolation (probabilistic mixing).
    """

    def __init__(self, in_f: int, out_f: int, kinds=("linear", "relu", "sigmoid")):
        super().__init__()
        self.branches = nn.ModuleDict()
        for kind in kinds:
            if kind == "linear":
                linear_layer = nn.Linear(in_f, out_f)
                with torch.no_grad():
                    # Initialize close to identity when possible
                    square = min(in_f, out_f)
                    linear_layer.weight.zero_()
                    linear_layer.weight[:square, :square] = torch.eye(square)
                self.branches[kind] = linear_layer
            else:
                activation = getattr(nn, {"relu": "ReLU", "sigmoid": "Sigmoid"}[kind])()
                self.branches[kind] = nn.Sequential(nn.Linear(in_f, out_f), activation)
        self.mix_logits = nn.Parameter(torch.zeros(len(kinds)))  # unconstrained
        self.kinds = kinds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.mix_logits, dim=0)
        output = 0
        for i, kind in enumerate(self.kinds):
            output = output + weights[i] * self.branches[kind](x)
        return output


class SkipPreserveBlock(nn.Module):
    """
    Insert new layer while preserving old direct connection (skip).
    """

    def __init__(self, in_f: int, hidden_f: int, out_f: int):
        super().__init__()
        self.new = CompoundNode(in_f, out_f, kinds=("linear", "relu"))
        self.skip = nn.Linear(in_f, out_f, bias=False)  # preserved old path
        with torch.no_grad():
            # Start harmless: zero the new linear branch and skip path
            linear_branch = self.new.branches["linear"]
            linear_branch.weight.zero_()
            if linear_branch.bias is not None:
                linear_branch.bias.zero_()
            self.skip.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.new(x) + self.skip(x)


class GradNormalizer:
    """
    Layer-by-layer gradient normalization using max-norm or L2 across parameters.
    Call after loss.backward(), before optimizer.step().
    """

    def __init__(self, modules, norm: str = "max", eps: float = 1e-8, target: float = 1.0):
        self.modules = list(modules)
        self.norm = norm
        self.eps = eps
        self.target = target

    def _norm(self, params):
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=params[0].device)
        if self.norm == "l2":
            total = torch.zeros((), device=grads[0].device)
            for g in grads:
                total = total + (g.detach() ** 2).sum()
            return torch.sqrt(total + 1e-12)
        # default: max
        return torch.stack([g.detach().abs().max() for g in grads]).max()

    @torch.no_grad()
    def apply(self):
        previous_layer_scale = None
        for module in self.modules:
            params = [p for p in module.parameters() if p.grad is not None]
            if not params:
                continue
            layer_scale = self._norm(params)
            if previous_layer_scale is not None and layer_scale > 0:
                scale = previous_layer_scale / (layer_scale + self.eps)
                clamped = torch.clamp(scale, 0.1, 10.0)
                for p in params:
                    p.grad.mul_(clamped)
            previous_layer_scale = layer_scale


class RNNBPTTNormalizer:
    """
    Time-wise normalization for unfolded RNNs/Transformers (blockwise).
    Provide a list of lists of modules: time_steps[t] = [layer1, layer2, ...]
    Assumes time_steps are provided in reverse (backprop-to-front) order.
    """

    def __init__(self, time_steps, norm: str = "max"):
        self.time_steps = time_steps
        self.norm = norm

    @torch.no_grad()
    def apply(self):
        for t_layers in self.time_steps:  # iterate in given (backwards) order
            layer_normer = GradNormalizer(t_layers, norm=self.norm)
            layer_normer.apply()