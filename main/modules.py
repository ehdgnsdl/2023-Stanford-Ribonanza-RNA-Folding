import admin_torch as admin
import torch
import torch.nn.functional as F
import xformers.ops as xops
from rotary_embedding_torch import RotaryEmbedding as Rope
from torch import Tensor, nn
from x_transformers.x_transformers import RelativePositionBias as RelPB
from apex.normalization import FusedLayerNorm, FusedRMSNorm


class ProjectOut(nn.Module):
    def __init__(
        self,
        p_dropout: float,
        d_model: int,
        aux_loop: bool,
        aux_struct: bool,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p_dropout)
        d = {"react": nn.Linear(d_model, 2)}
        d = d | ({"loop": nn.Linear(d_model, 7)} if aux_loop else {})
        d = d | ({"struct": nn.Linear(d_model, 3)} if aux_struct else {})
        self.linears = nn.ModuleDict(d)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        out = {k: v(x) for k, v in self.linears.items()}
        return out


class RNA_Model(nn.Module):
    def __init__(
        self,
        pos_rope: bool,
        pos_bias_heads: int,
        pos_bias_params: tuple[int, int],
        norm_rms: str,
        norm_lax: bool,
        emb_grad_frac: float,
        n_mem: int,
        aux_loop: str | None,
        aux_struct: str | None,
        kernel_size_gc: int,
        **kwargs,
    ):
        super().__init__()
        global Norm
        Norm = get_layer_norm(FusedRMSNorm if norm_rms else FusedLayerNorm, norm_lax)
        d_model = (d_heads := kwargs["d_heads"]) * (n_heads := kwargs["n_heads"])
        p_dropout, n_layers = kwargs["p_dropout"], kwargs["n_layers"]
        layers = [EncoderLayer(**kwargs) for i in range(n_layers)]
        self.layers = nn.ModuleList(layers)
        self.mem = nn.Parameter(torch.randn(1, n_mem, d_model)) if n_mem else None
        self.emb = nn.Embedding(5, d_model, 0)
        self.rope = Rope(d_heads, seq_before_head_dim=True) if pos_rope else None
        self.pos_bias = None
        if pos_bias_heads:
            assert (heads := pos_bias_heads) <= n_heads
            self.pos_bias = RelPB(d_heads**0.5, False, *pos_bias_params, heads)
        self.out = ProjectOut(p_dropout, d_model, aux_loop != None, aux_struct != None)
        self.res = kwargs["norm_layout"] == "dual"
        self.emb_grad = emb_grad_frac
        

    def forward(self, x0: Tensor) -> Tensor:
        seq = x0['seq']
        mask = x0['mask']
        bpps = x0['As'] 
                
        x = self.emb(seq)        
        b = bpps[:, :, :, 0]

        if self.mem != None:
            mask = F.pad(mask, (self.mem.size(1), 0))
            x = torch.cat([self.mem.expand(x.size(0), -1, -1), x], 1)
        if 1 > self.emb_grad > 0:
            x = x * self.emb_grad + x.detach() * (1 - self.emb_grad)
        res = x * self.layers[0].res_scale if self.res else None
        for f in self.layers:
            if self.res:                
                x, res = f(x, b, res, mask, self.rope, self.pos_bias)                
            else:
                x = f(x, mask, self.rope, self.pos_bias)            
        x = x[:, self.mem.size(1) :] if self.mem != None else x
            
        return self.out(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,        
        d_heads: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
        ffn_multi: int,
        ffn_bias: bool,
        qkv_bias: bool,
        att_fn: str,
        norm_layout: str,        
        **kwargs,
    ):
        super().__init__()
        d_ffn = (d_model := d_heads * n_heads) * ffn_multi
        
        self.att = Multi_head_Attention(d_model, n_heads, p_dropout, qkv_bias, att_fn)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn, ffn_bias),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(d_ffn, d_model, ffn_bias),
            nn.Dropout(p_dropout),
        )
        if norm_layout == "dual":
            self.forward = self.forward_dual
            self.norm = nn.ModuleList([Norm(d_model) for _ in range(4)])
            self.res_scale = 0.1
        elif norm_layout == "sandwich":
            self.forward = self.forward_sandwich
            self.norm = nn.ModuleList([Norm(d_model) for _ in range(4)])
        else:
            self.forward = self.forward_post
            self.norm = nn.ModuleList([Norm(d_model) for _ in range(2)])
            self.res = nn.ModuleList([admin.as_module(n_layers * 2) for _ in range(2)])
                    
        self.gru = nn.GRU(d_model, d_model//2, kwargs['n_layers_lstm'], batch_first = True, bidirectional = True, dropout = p_dropout)

        kernel_size_gc = 7
        self.attn3 = ResidualBPPAttention(d_model, kernel_size=kernel_size_gc, dropout=p_dropout)    
        

    def forward_post(self, x: Tensor, *args, **kwargs):
        x = self.norm[0](self.res[0](x, self.att(x, *args, **kwargs)))
        x = self.norm[1](self.res[1](x, self.ffn(x)))
        return x

    def forward_sandwich(self, x: Tensor, *args, **kwargs):
        x = x + self.norm[1](self.att(self.norm[0](x), *args, **kwargs))
        x = x + self.norm[3](self.ffn(self.norm[2](x)))
        return x

    def forward_dual(self, x: Tensor, b: Tensor, res: Tensor, *args, **kwargs):
        
        x_att = self.att(x, *args, **kwargs)
        res = res + x_att * self.res_scale
        x = self.norm[0](x + x_att) + self.norm[1](res)
        x_ffn = self.ffn(x)
        res = res + x_ffn * self.res_scale
        x = self.norm[2](x + x_ffn) + self.norm[3](res)

        # for bpps
        x = x.permute([0, 2, 1])  # [batch, d-emb, seq]
        x = self.attn3(x, b)
        x = x.permute([0, 2, 1])  # [batch, d-emb, seq]                
                    
        x, _ = self.gru(x) # by junseong
                
        return x, res

class ResidualBPPAttention(nn.Module):
    def __init__(self, d_model:int, kernel_size:int, dropout:float):
        super().__init__()
        self.conv1 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.conv2 = Conv(d_model, d_model, kernel_size=kernel_size, dropout=dropout)
        self.relu = nn.ReLU()

    def forward(self, src, attn):
        h = self.conv2(self.conv1(torch.bmm(src, attn)))
        return self.relu(src + h)

class Conv(nn.Module):
    def __init__(self, d_in:int, d_out:int, kernel_size:int, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(d_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        return self.dropout(self.relu(self.bn(self.conv(src))))

class Multi_head_Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        p_dropout: float,
        qkv_bias: bool,
        att_fn: str,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_heads = d_model // n_heads
        self.p_dropout = p_dropout
        self.qkv = nn.Linear(d_model, d_model * 3, qkv_bias)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(p_dropout),
        )
        self.att = self.xmea if att_fn == "xmea" else self.sdpa

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None,
        rope: nn.Module | None,
        pos_bias: nn.Module | None,
    ) -> Tensor:
        B, L, N, D = (*x.shape[:2], self.n_heads, self.d_heads)
        qkv = [_.view(B, L, N, D) for _ in self.qkv(x).chunk(3, -1)]
        if rope != None:
            qkv[0] = rope.rotate_queries_or_keys(qkv[0])
            qkv[1] = rope.rotate_queries_or_keys(qkv[1])
        bias = None
        if pos_bias != None:
            bias = pos_bias(L, L).to(qkv[0].dtype)
            bias = bias.unsqueeze(0).expand(B, -1, -1, -1)
            if N > bias.size(1):
                bias = F.pad(bias, (*[0] * 5, N - bias.size(1)))
        if mask != None:
            mask = self.mask_to_bias(mask, qkv[0].dtype)
            bias = mask if bias == None else bias + mask
        x = self.att(qkv, bias.contiguous()).reshape(B, L, N * D)
        return self.out(x)

    def sdpa(self, qkv: tuple[Tensor, Tensor, Tensor], bias: Tensor | None) -> Tensor:
        p_drop = self.p_dropout if self.training else 0
        qkv = [_.transpose(1, 2) for _ in qkv]
        x = F.scaled_dot_product_attention(*qkv, bias, p_drop)
        return x.transpose(1, 2)

    def xmea(self, qkv: tuple[Tensor, Tensor, Tensor], bias: Tensor | None) -> Tensor:
        p_drop = self.p_dropout if self.training else 0
        if bias != None and (L := qkv[0].size(1)) % 8:
            pad = -(L // -8) * 8 - L
            bias = F.pad(bias, (0, pad, 0, pad)).contiguous()[..., :L, :L]
        x = xops.memory_efficient_attention(*qkv, bias, p_drop)
        return x

    def mask_to_bias(self, mask: Tensor, float_dtype: torch.dtype) -> Tensor:
        mask = mask.to(float_dtype) - 1
        mask[mask < 0] = float("-inf")
        mask = mask.view(mask.size(0), 1, 1, mask.size(-1))
        mask = mask.expand(-1, self.n_heads, mask.size(-1), -1)
        return mask


def get_layer_norm(cls: nn.Module, norm_lax: bool) -> nn.Module:
    class LayerNorm(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.forward = super().forward
            if norm_lax:
                self.lax = torch.compile(lambda x: x / x.max())
                self.forward = self.forward_lax

        def forward_lax(self, x):
            return super().forward(self.lax(x))

    return LayerNorm
