import torch
import torch.nn.functional as F
from torch import nn
from reformer_pytorch import LSHSelfAttention
from functools import partial


class BasicTransformer(nn.Module):
    """
    ### Vanilla Transformer Layer with $O(N^2)$ complexity for self-attention
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float=0.1,
                 use_flash_attention: bool=False, is_gated: bool=False):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        """
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.attn = SelfAttention(d_model, n_heads, d_head, 
                                  use_flash_attention=use_flash_attention)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, is_gated=is_gated)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, seq_len, d_model]`
        """
        # Self attention
        x = self.dropout(self.attn(self.norm1(x))) + x
        # Feed-forward network
        x = self.dropout(self.ff(self.norm2(x))) + x
        #
        return x


class SelfAttention(nn.Module):
    """
    ### Self Attention Layer
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, is_inplace: bool=False, use_flash_attention: bool=False):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of a attention head
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        :param use_flash_attention: specifies whether to use the flash attention
        """
        super().__init__()

        self.is_inplace = is_inplace
        self.use_flash_attention = use_flash_attention
        self.n_heads = n_heads
        self.d_head = d_head

        # Attention scaling factor
        self.scale = d_head ** -0.5

        # Query, key and value mappings
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_model, d_attn, bias=False)
        self.to_v = nn.Linear(d_model, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))

        # Setup [flash attention](https://github.com/HazyResearch/flash-attention).
        # Flash attention is only used if it's installed
        # and `CrossAttention.use_flash_attention` is set to `True`.
        try:
            # You can install flash attention by cloning their Github repo,
            # [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
            # and then running `python setup.py install`
            from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
            self.flash = partial(flash_attn_qkvpacked_func, softmax_scale=self.scale)
        # Set to `None` if it's not installed
        except ImportError:
            print("Flash Attention import failed. Make sure you have installed it.")
            self.flash = None

    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, seq_len, d_model]`
        """

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Use flash attention if it's available and the head size is less than or equal to `128`
        if self.use_flash_attention and self.flash is not None and self.d_head <= 128:
            return self.flash_attention(q, k, v)
        # Otherwise, fallback to normal attention
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash Attention

        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Get batch size and number of elements along sequence axis (`width * height`)
        batch_size, seq_len, _ = q.shape

        # Stack `q`, `k`, `v` vectors for flash attention, to get a single tensor of
        # shape `[batch_size, seq_len, 3, n_heads * d_head]`
        qkv = torch.stack((q, k, v), dim=2)
        # Split the heads
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        # Flash attention works for head sizes `32`, `64` and `128`, so we have to pad the heads to
        # fit this size.
        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f'Head size ${self.d_head} too large for Flash Attention')

        # Pad the heads
        if pad:
            qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1)

        # Compute attention
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        # This gives a tensor of shape `[batch_size, seq_len, n_heads, d_padded]`
        out = self.flash(qkv.half()) # Use half precision for flash attention
        out = out.float() # Convert back to float precision
        # Truncate the extra head size
        out = out[:, :, :, :self.d_head]
        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)

        # Map to `[batch_size, seq_len, d_model]` with a linear layer
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention
        
        :param q: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param k: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        :param v: are the query vectors before splitting heads, of shape `[batch_size, seq, d_attn]`
        """

        # Split them to heads of shape `[batch_size, seq_len, n_heads, d_head]`
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        # Calculate attention $\frac{Q K^\top}{\sqrt{d_{key}}}$
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
        # q = q.permute(0, 2, 1, 3)  # [batch_size, n_heads, seq_len, d_head]
        # k = k.permute(0, 2, 3, 1)  # [batch_size, n_heads, d_head, seq_len]
        # attn = torch.matmul(q, k) * self.scale  # [batch_size, n_heads, seq_len, seq_len]

        # Compute softmax
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$$
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        # Compute attention output
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$$
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        # v = v.permute(0, 2, 1, 3)  # [batch_size, n_heads, seq_len, d_head]
        # out = torch.matmul(attn, v)  # [batch_size, n_heads, seq_len, d_head]
        # out = out.permute(0, 2, 1, 3)
        
        # Reshape to `[batch_size, seq_len, n_heads * d_head]`
        out = out.reshape(*out.shape[:2], -1)
        # Map to `[batch_size, seq_len, d_model]` with a linear layer
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self, d_model: int, d_mult: int = 4, is_gated: bool=False):
        """
        :param d_model: is the input embedding size
        :param d_mult: is multiplicative factor for the hidden layer size
        """
        super().__init__()
        if is_gated:
            self.net = nn.Sequential(
                GeGLU(d_model, d_model * d_mult),
                nn.Dropout(0.),
                nn.Linear(d_model * d_mult, d_model),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model * d_mult),
                nn.GELU(),
                nn.Dropout(0.),
                nn.Linear(d_model * d_mult, d_model),
            )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)


class LinearAttention(nn.Module):
    """
    ### Linear Attention Layer
    """
    def __init__(self, d_model: int, map_name: str="elu+1", eps=1e-6):
        super().__init__()
        self.eps = eps
        if map_name == "elu+1":
            self.map_func = lambda x: F.elu(x) + 1
        else:
            raise NotImplementedError(f"Mapping function {map_name} not implemented")
        self.eps = eps
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
    
    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, seq_len, d_model]`
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self.map_func(q)
        k = self.map_func(k)
        KV = torch.einsum("bld,blh->bhd", k, v) # KV matrix
        Z = 1/(torch.einsum("bld,bd->bl", q, k.sum(dim=1)) + self.eps) # Normalization factor
        V = torch.einsum("bld,bhd,bl->blh", q, KV, Z)
        return V.contiguous()
    

class LinearTransformer(nn.Module):
    """
    ### Linear Transformer Layer with $O(N)$ complexity for self-attention
    """
    def __init__(self, d_model: int, map_name: str="elu+1", eps=1e-6, dropout=0.1):
        super().__init__()
        self.attn = LinearAttention(d_model, map_name, eps)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, seq_len, d_model]`
        """
        x = self.dropout(self.attn(self.norm1(x))) + x
        x = self.dropout(self.ff(self.norm2(x))) + x
        return x


class SparseTransformer(nn.Module):
    """
    ### Sparse Transformer Layer with $O(N*logN)$ complexity for self-attention
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float=0.1):
        super().__init__()
        self.attn = LSHSelfAttention(d_model, heads=1, n_hashes=2, bucket_size=50)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, seq_len, d_model]`
        """
        x = self.dropout(self.attn(self.norm1(x))) + x
        x = self.dropout(self.ff(self.norm2(x))) + x
        return x
