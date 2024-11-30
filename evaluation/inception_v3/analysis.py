import torch
from ..counts import learnable_param_count, flop_backward
from .base import build_model

def main():
    mod = build_model(10, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'num params: {learnable_param_count(mod)}')
    print(f'bwd flops: {flop_backward(mod, (10, 3, 299, 299))}')

if __name__ == '__main__':
    main()
