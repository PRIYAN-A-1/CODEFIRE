from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 384
    num_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1024
    max_context_length: int = 512
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.
    name: str = 'PRIYAN-CODE-23M'
    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0
        assert (self.hidden_size // self.num_attention_heads) % 2 == 0
        assert self.max_context_length > 0 and self.vocab_size > 0

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight

def rope(x, offset, theta):
    d = x.shape[-1]
    freq = 1. / theta ** (torch.arange(0, d, 2, device=x.device).float() / d)
    phase = torch.outer(torch.arange(offset, offset + x.shape[-2], device=x.device).float(), freq)
    cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a*cos-b*sin, a*sin+b*cos), dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c=c
        self.qkv=nn.Linear(c.hidden_size,3*c.hidden_size,bias=False)
        self.out=nn.Linear(c.hidden_size,c.hidden_size,bias=False)
    def forward(self,x,cache=None,use_cache=False):
        b,t,d=x.shape; h=self.c.num_attention_heads
        q,k,v=[z.view(b,t,h,d//h).transpose(1,2) for z in self.qkv(x).chunk(3,-1)]
        offset=0 if cache is None else cache[0].shape[-2]
        q,k=rope(q,offset,self.c.rope_theta),rope(k,offset,self.c.rope_theta)
        if cache is not None:
            k=torch.cat((cache[0],k),-2);v=torch.cat((cache[1],v),-2)
        mask=None
        if offset:
            mask=torch.arange(k.shape[-2],device=x.device)[None,:] <= torch.arange(offset,offset+t,device=x.device)[:,None]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,is_causal=not offset)
        return self.out(y.transpose(1,2).contiguous().view(b,t,d)), (k,v) if use_cache else None

class Block(nn.Module):
    def __init__(self,c):
        super().__init__();self.norm1=RMSNorm(c.hidden_size);self.norm2=RMSNorm(c.hidden_size);self.attn=Attention(c)
        self.gate=nn.Linear(c.hidden_size,c.intermediate_size,bias=False)
        self.up=nn.Linear(c.hidden_size,c.intermediate_size,bias=False)
        self.down=nn.Linear(c.intermediate_size,c.hidden_size,bias=False)
    def forward(self,x,cache=None,use_cache=False):
        a,kv=self.attn(self.norm1(x),cache,use_cache);x=x+a;z=self.norm2(x)
        return x+self.down(F.silu(self.gate(z))*self.up(z)),kv

class Transformer(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c;self.gradient_checkpointing=False
        self.embed=nn.Embedding(c.vocab_size,c.hidden_size)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm=RMSNorm(c.hidden_size);self.head=nn.Linear(c.hidden_size,c.vocab_size,bias=False)
        if c.tie_word_embeddings:self.head.weight=self.embed.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight, std=.02)
    def forward(self,ids,labels=None,cache=None,use_cache=False):
        n=ids.shape[1]+(cache[0][0].shape[-2] if cache else 0)
        if n>self.config.max_context_length:raise ValueError('Context window exceeded')
        x=self.embed(ids);new=[]
        for i,block in enumerate(self.blocks):
            if self.training and self.gradient_checkpointing:
                x=checkpoint(lambda z, layer=block: layer(z)[0],x,use_reentrant=False);kv=None
            else:x,kv=block(x,cache[i] if cache else None,use_cache)
            new.append(kv)
        logits=self.head(self.norm(x))
        loss=None if labels is None else F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),labels.reshape(-1),ignore_index=-100)
        return logits,loss,new if use_cache else None
    def parameter_count(self):return sum(p.numel() for p in self.parameters())
