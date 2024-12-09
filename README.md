# model-calculator
calculator for pytorch model

## Attention
1. MHA/MQA/QGA

```sh
python mha.py

# it will show:
# test MHA：
# input size: torch.Size([128, 1024, 8192])
# output size: torch.Size([128, 1024, 8192])
# cost_time: 12.70s
# test MQA：
# input size: torch.Size([128, 1024, 8192])
# output size: torch.Size([128, 1024, 8192])
# cost_time: 7.94s
# test GQA：
# input size: torch.Size([128, 1024, 8192])
# output size: torch.Size([128, 1024, 8192])
# cost_time: 8.35s
```

