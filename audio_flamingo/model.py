import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from torch.autograd import Function
from zeta.nn import audio_to_text, Attention
from zeta.structs import Transformer, Decoder, AutoregressiveWrapper

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(numer, denom):
    return (numer % denom) == 0


# distributed


def pad_dim_to(t, length, dim=0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))


def all_gather_variable_batch(t):
    device, rank, world_size = (
        t.device,
        dist.get_rank(),
        dist.get_world_size(),
    )

    size = torch.tensor(t.shape[0], device=device, dtype=torch.long)
    sizes = [
        torch.empty_like(size, device=device, dtype=torch.long)
        for i in range(world_size)
    ]
    dist.all_gather(sizes, size)

    sizes = torch.stack(sizes)
    max_size = sizes.amax().item()

    padded_t = pad_dim_to(t, max_size, dim=0)
    gathered_tensors = [
        torch.empty_like(
            padded_t, device=device, dtype=padded_t.dtype
        )
        for i in range(world_size)
    ]
    dist.all_gather(gathered_tensors, padded_t)

    gathered_tensor = torch.cat(gathered_tensors)
    seq = torch.arange(max_size, device=device)

    mask = rearrange(seq, "j -> 1 j") < rearrange(sizes, "i -> i 1")
    mask = rearrange(mask, "i j -> (i j)")

    gathered_tensor = gathered_tensor[mask]
    sizes = sizes.tolist()

    return gathered_tensor, sizes


class AllGather(Function):
    @staticmethod
    def forward(ctx, x):
        assert dist.is_initialized() and dist.get_world_size() > 1
        x, batch_sizes = all_gather_variable_batch(x)
        ctx.batch_sizes = batch_sizes
        return x

    @staticmethod
    def backward(ctx, grads):
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        grads_by_rank = grads.split(batch_sizes, dim=0)
        return grads_by_rank[rank]


all_gather = AllGather.apply


# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# to latents


class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(
            max_seq_len, device=device, dtype=self.inv_freq.dtype
        )
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = (
            LayerNorm(context_dim) if norm_context else nn.Identity()
        )

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = (
            nn.Sequential(
                nn.Linear(dim, ff_inner_dim * 2, bias=False),
                SwiGLU(),
                nn.Linear(ff_inner_dim, dim, bias=False),
            )
            if parallel_ff
            else None
        )

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge and combine heads

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out


class XCAttention(nn.Module):
    """
    Cross-Correlation Attention module.

    Args:
        dim (int): The input dimension.
        dim_head (int): The dimension of each attention head.
        heads (int): The number of attention heads.
        dropout (int): The dropout rate.
        context_dim (int): The dimension of the context.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        heads (int): The number of attention heads.
        scale (float): The scaling factor for attention scores.
        cross_attn (CrossAttention): The cross-attention module.
        tanh (nn.Tanh): The Tanh activation function.
        dense (nn.Linear): The dense layer.
        norm (LayerNorm): The layer normalization module.
        dropout (nn.Dropout): The dropout module.

    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        dropout: int,
        context_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.dropout = dropout
        self.heads = heads
        dim_head * heads
        self.scale = dim_head**-0.5

        self.cross_attn = CrossAttention(
            dim,
            context_dim=context_dim,
            dim_head=dim_head,
            heads=heads,
            parallel_ff=True,
            ff_mult=4,
            norm_context=True,
        )

        # Tanh
        self.tanh = nn.Tanh()

        # Dense layer
        self.dense = nn.Linear(dim, dim)

        # LayerNorm
        self.norm = LayerNorm(dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        # b, s = x.shape
        # x = audio_to_text(x, seqlen=s, dim=self.dim)
        # print(x.shape)

        skip = x

        # Cross Attention
        x = self.cross_attn(x, x)

        # LayerNorm
        x = self.norm(x)

        # Tanh
        x = self.tanh(x) + skip

        # Dense layer
        x = self.dense(x)

        return self.tanh(x)


class AudioFlamingoEncoderBlock(nn.Module):
    """
    AudioFlamingoEncoderBlock is a module that represents a single block of the AudioFlamingo encoder.
    It applies multiple layers of attention and transformation to the input audio representation.

    Args:
        dim (int): The dimension of the input audio representation.
        heads (int): The number of attention heads.
        depth (int): The number of layers in the encoder block.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        context_dim (int): The dimension of the context vector.
        device (str): The device to run the module on.

    Attributes:
        dim (int): The dimension of the input audio representation.
        heads (int): The number of attention heads.
        depth (int): The number of layers in the encoder block.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        context_dim (int): The dimension of the context vector.
        device (str): The device to run the module on.
        gated_xtten_layers (nn.ModuleList): List of Gated XAttention Dense layers.
        norm (LayerNorm): Layer normalization module.
        rpl_layers (nn.ModuleList): List of Representation transformation layers.

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: int,
        context_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dim_head = dim_head
        self.dropout = dropout
        self.context_dim = context_dim
        # self.device = device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Gated XAttention Dense
        self.gated_xtten_layers = nn.ModuleList([])
        self.gated_xtten_layers.append(
            XCAttention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout=dropout,
                context_dim=context_dim,
                *args,
                **kwargs,
            )
        )

        # LayerNorm
        self.norm = LayerNorm(dim)

        # Representation transformation layers
        self.rpl_layers = nn.ModuleList([])
        for i in range(3):
            self.rpl_layers.append(
                Attention(
                    dim,
                    dim_head,
                    heads,
                    causal=True,
                    qk_norm=True,
                    *args,
                    **kwargs,
                )
            )

    def forward(self, x: Tensor):
        """
        Forward pass of the AudioFlamingoEncoderBlock.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        b, s = x.shape
        x = audio_to_text(x, seqlen=s, dim=self.dim)

        skip = x

        # Audio Representation layers
        for layer in self.rpl_layers:
            x, _ = layer(x)

        # Gated XAttn Dense
        for layer in self.gated_xtten_layers:
            x = layer(x)

        # LayerNorm
        x = self.norm(x)

        # Skip connection
        x = x + skip

        return x


class AudioFlamingo(nn.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        max_seq_len: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: float,
        context_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.heads = heads
        self.depth = depth
        self.dim_head = dim_head
        self.dropout = dropout
        self.context_dim = context_dim

        self.transformer = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            emb_dim=dim,
            post_emb_norm=True,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                rotary_xpos=True,
                attn_flash=True,
            ),
        )

        self.decoder = AutoregressiveWrapper(self.transformer)

        # AudioFlamingoEncoderBlock layers
        self.af_blocks = nn.ModuleList([])
        self.af_blocks.append(
            AudioFlamingoEncoderBlock(
                dim=dim,
                heads=heads,
                depth=depth,
                dim_head=dim_head,
                dropout=dropout,
                context_dim=context_dim,
                *args,
                **kwargs,
            )
        )

        # LayerNorm
        self.norm = LayerNorm(dim)

    def forward(self, text: Tensor, audio: Tensor):
        # Text shape - (b, s, d)
        # Audio shape - (b, s)

        # Apply audio blocks to audio
        for block in self.af_blocks:
            audio = block(audio)
            audio = self.norm(audio)

        # Transformer
        return self.decoder(text, context=audio)
