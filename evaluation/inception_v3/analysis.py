import torch
from ..counts import learnable_param_count, flop_backward, flop_forward
from .train import build_model

def main():
    base_model = build_model(True, 10, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print('base')
    print(f'num params: {learnable_param_count(base_model)}')
    print(f'fwd flops: {flop_forward(base_model, (10, 3, 299, 299))}')
    print(f'fwd flops: {flop_backward(base_model, (10, 3, 299, 299))}')
    pd_model = build_model(False, 10, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print('with predefined')
    print(f'num params: {learnable_param_count(pd_model)}')
    print(f'fwd flops: {flop_forward(pd_model, (10, 3, 299, 299))}')
    print(f'bwd flops: {flop_backward(pd_model, (10, 3, 299, 299))}')

if __name__ == '__main__':
    main()
