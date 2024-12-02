import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from typing import Tuple

def learnable_param_count(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)

def flop_forward(mod, input_size: Tuple):
    istrain = mod.training
    mod.eval()

    input = torch.randn(input_size)

    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        _ = mod(input)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        mod.train()
    return total_flops


def flop_backward(mod, input_size: Tuple):
    istrain = mod.training
    mod.eval()

    input = torch.randn(input_size)

    flop_counter = FlopCounterMode(display=False)
    loss = mod(input).sum()
    with flop_counter:
        loss.backward()
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        mod.train()
    return total_flops

