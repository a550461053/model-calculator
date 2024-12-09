import os
import pdb
import pickle
import time

import torch

class Attention(torch.nn.Module):
    def __init__(self, query, key, value, attention_mask=None, dropout=0.1):
        super(Attention, self).__init__()
        self.query = query
        self.key = key
        self.value = value
        self.attention_mask = attention_mask
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, query, key, value):
        # query: [batch_size, seq_len, d_model]
        # key: [batch_size, seq_len, d_model]
        # value: [batch_size, seq_len, d_model]
        # attention_mask: [batch_size, seq_len, seq_len]
        # return: [batch_size, seq_len, d_model]
        d_model = query.size(-1)
        # [b,s,h] x [b,h,s] -> [b,s,s]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_model ** 0.5)
        if self.attention_mask is not None:
            scores = scores.masked_fill(self.attention_mask == 0, -1e9)
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        # [b,s,s] x [b,s,h] -> [b,s,h]
        output = torch.matmul(attention, value)
        return output

class MHA(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MHA, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_model = hidden_size // num_heads
        self.query_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.key_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.value_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_state, attention_mask=None):
        # hidden_state: [batch_size, seq_len, d_model]
        # attention_mask: [batch_size, seq_len, seq_len]
        # return: [batch_size, seq_len, d_model]
        
        batch_size = hidden_state.size(0)
        seq_len = hidden_state.size(1)
        # [b,s,d] -> [b,s,d]
        query = self.query_proj(hidden_state)
        key = self.key_proj(hidden_state)
        value = self.value_proj(hidden_state)
        
        # [b,s,d] -> [b,s,n,d] -> [b,n,s,d]
        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)
        
        # # split hidden_size into num_heads
        # # [b,s,d] -> [b,n,s,d] -> [b,s,n,d]
        # query = self.query_proj(query).view(batch_size, self.num_heads, seq_len, self.d_model).transpose(1, 2)
        # key = self.key_proj(key).view(batch_size, self.num_heads, seq_len, self.d_model).transpose(1, 2)
        # value = self.value_proj(value).view(batch_size, self.num_heads, seq_len, self.d_model).transpose(1, 2)
        
        # [b,n,s,d] x [b,n,d,s] -> [b,n,s,s]
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.d_model ** 0.5)
        
        scores = torch.softmax(scores, dim=-1)
        
        # [b,n,s,s] x [b,n,s,d] -> [b,n,s,d]
        output = torch.matmul(scores, value)
        # [b,n,s,d] -> [b,s,n,d] -> [b,s,d]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        # [b,s,d] -> [b,s,d]
        
        output = self.out_proj(output)
        
        # attention = Attention(query, key, value, attention_mask)
        # output = attention(query, key, value)
        return output

    def split_head(self, tensor):
        # tensor: [batch_size, seq_len, hidden_size]
        # return: [batch_size, num_heads, seq_len, d_model]
        return tensor.view(tensor.size(0), tensor.size(1), self.num_heads, self.d_model).transpose(1, 2)


class MQA(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        # MQA: 多个head之间共享同一份K/V矩阵，只是Q（multi-query）
        # 优点：减少K/V矩阵的计算量和参数量，降低kvcache的显存占用
        # 缺点：精度损失
        # 样例：ChatGLM2
        super(MQA, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.key_proj = torch.nn.Linear(hidden_size, self.head_dim)
        self.value_proj = torch.nn.Linear(hidden_size, self.head_dim)
        self.output_proj = torch.nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_state):
        bs = hidden_state.size(0)
        seq_len = hidden_state.size(1)
        
        query = self.query_proj(hidden_state)  # [bs, seq_len, hidden_size]
        key = self.key_proj(hidden_state)      # [bs, seq_len, head_dim]
        value = self.value_proj(hidden_state)  # [bs, seq_len, head_dim]
        
        query = self.split_head(query)  # [bs, num_heads, seq_len, head_dim]
        key = self.split_head(key, 1)   # [bs, 1, seq_len, head_dim]
        value = self.split_head(value, 1)   # [bs, 1, seq_len, head_dim]
        
        # [bs, num_heads, seq_len, head_dim] x [bs, 1, head_dim, seq_len] -> [bs, num_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        scores = torch.softmax(scores, dim=-1)
        
        # [bs, num_heads, seq_len, seq_len] x [bs, 1, seq_len, head_dim] -> [bs, num_heads, seq_len, head_dim]
        output = torch.matmul(scores, value)
        
        # [bs, num_heads, seq_len, head_dim] -> [bs, seq_len, num_heads, head_dim] -> [bs, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.hidden_size)
        
        # [bs, seq_len, hidden_size] -> [bs, seq_len, hidden_size]
        output = self.output_proj(output)
        
        return output
        
        
    def split_head(self, tensor, head_num=None):
        # tensor: [batch_size, seq_len, hidden_size]
        # return: [batch_size, num_heads, seq_len, head_dim]
        head_num = head_num if head_num is not None else self.num_heads
        return tensor.view(tensor.size(0), -1, head_num, self.head_dim).transpose(1, 2)

class GQA(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, num_groups):
        # GQA: 将多个Q进行分组，每个组内共享K/V
        #   group=1时，等价于MQA，所有查询头只有一个K/V
        #   group=num_heads时，等价于MHA，每个查询头都有一个K/V
        super(GQA, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = hidden_size // num_heads
        self.query_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.key_proj = torch.nn.Linear(hidden_size, self.head_dim * self.num_groups)
        self.value_proj = torch.nn.Linear(hidden_size, self.head_dim * self.num_groups)
        self.output_proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state):
        bs = hidden_state.size(0)
        seq_len = hidden_state.size(1)

        query = self.query_proj(hidden_state)  # [bs, seq_len, hidden_size]
        key = self.key_proj(hidden_state)   # [bs, seq_len, head_dim * num_groups]
        value = self.value_proj(hidden_state)  # [bs, seq_len, head_dim * num_groups]
        
        query = self.split_head(query)  # [b,n,s,d]
        key = self.split_head(key, group_num=self.num_groups)  # [b,n,s,d]
        value = self.split_head(value, group_num=self.num_groups)  # [b,n,s,d]
        
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(scores, value)
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.hidden_size)
        
        output = self.output_proj(output)
        return output
        
    def split_head(self, tensor, head_num=None, group_num=None):
        # tensor: [batch_size, seq_len, hidden_size]
        # return: [batch_size, num_heads, seq_len, head_dim]
        bs, seq_len = tensor.size(0), tensor.size(1)
        num_heads = head_num if head_num is not None else self.num_heads
        if group_num is None:
            return tensor.view(bs, seq_len, num_heads, self.head_dim).transpose(1, 2)
        else:
            # [b,s,g,d]
            tensor = tensor.view(bs, seq_len, group_num, self.head_dim).transpose(1, 2)
            # [b,g,s,d]
            tensor = tensor[:, :, None, :, :].expand(bs, group_num, self.num_heads // group_num, seq_len, self.head_dim)
            # [b,g,n_g,s,d]
            tensor = tensor.reshape(bs, self.num_heads, seq_len, self.head_dim)
            # [b,n,s,d]
            return tensor
        

def test_mha(attn='MHA'):
    bs = 128
    seq_len = 1024
    hidden_size = 8192
    num_heads = 8
    num_groups = 2
    
    # inputs
    hidden_state = torch.randn(bs, seq_len, hidden_size)
    
    # test
    t1 = time.time()
    if attn == 'GQA':
        mha = eval(attn)(hidden_size, num_heads, num_groups=num_groups)
    else:
        mha = eval(attn)(hidden_size, num_heads)
    output = mha(hidden_state)
    t2 = time.time()
    print(f"test {attn}：")
    print(f"  input size: {hidden_state.size()}")
    print(f"  output size: {output.size()}")
    print(f"  cost_time: {t2-t1:.2f}s")
    

if __name__ == "__main__":
    test_mha("MHA")
    test_mha("MQA")
    test_mha("GQA")
