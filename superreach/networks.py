"""Network layers and real-observation normalization."""
import torch
from torch import nn


class RunningMeanStd(nn.Module):
    """Population moments from real observations, never from WM rollouts."""

    def __init__(self, width):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('variance', torch.ones(width))
        self.register_buffer('count', torch.zeros((), dtype=torch.int64))

    @torch.no_grad()
    def update(self, x):
        variance, mean = torch.var_mean(x, dim=0, correction=0)
        self.count.add_(x.shape[0])
        weight = x.shape[0] / self.count.to(x.dtype)
        delta = mean - self.mean
        self.variance.copy_((1-weight)*self.variance + weight*variance + weight*(1-weight)*delta.square())
        self.mean.add_(weight*delta)

    def forward(self, x):
        # A std floor bounds sensitivity even on nearly constant HOLD features.
        return ((x-self.mean)*self.variance.clamp_min(.25**2).rsqrt()).clamp(-5., 5.)


def build_mlp(
  input_dim: int,
  hidden_dims: tuple[int, ...],
  output_dim: int,
) -> nn.Sequential:
  """Build an MLP: ``input_dim → (Linear → LN → GELU)* → Linear(output_dim)``."""
  layers: list[nn.Module] = []
  curr = input_dim
  for h in hidden_dims:
    layers += [nn.Linear(curr, h), nn.LayerNorm(h), nn.GELU()]
    curr = h
  layers.append(nn.Linear(curr, output_dim))
  return nn.Sequential(*layers)
