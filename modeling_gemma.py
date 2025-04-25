import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache:
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the key and value caches with new key and value states.

        Args:
            key_states (torch.Tensor): New key states to be added to the cache.
            value_states (torch.Tensor): New value states to be added to the cache.
            layer_idx (int): Index of the current layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value states.
        """
        if len(self.key_cache) <= layer_idx:
            # if we never added anything to the cache yet, we just add the current key and value states
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # if we have already added something to the cache, we need to concatenate the new key and value states to the existing key and value states
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)

        # ...and then we return the updated key and value states
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
        
class GemmaConfig():
    """
    Configuration class for the Gemma language model.

    This class holds configuration parameters for the Gemma language model, which is a decoder-only model.

    1. vocab_size : Size of the vocabulary (number of tokens) the model can understand.
    2. hidden_size : Dimension of the hidden layers throughout the model.
    3. intermediate_size : Dimension of the feed-forward network's intermediate layer.
    4. num_hidden_layers : Number of transformer layers in the model.
    5. num_attention_heads : Number of attention heads for multi-head attention.
    6. num_key_value_heads : Number of key-value heads (for grouped-query attention).
    7. head_dim (default: 256): Dimension of each attention head.
    8. max_position_embeddings (default: 8192): Maximum sequence length the model can handle.
    9. rms_norm_eps (default: 1e-6): Small constant for numerical stability in RMSNorm.
    10. rope_theta (default: 10000.0): Base value for rotary position embeddings.
    11. attention_bias (default: False): Whether to use bias terms in attention calculations.
    12. attention_dropout (default: 0.0): Dropout probability for attention weights.
    13. pad_token_id (default: None): Token ID used for padding sequences.

    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim = 256,
        max_position_embeddings = 8192,
        rms_norm_eps = 1e-6,
        rope_theta = 10000.0,
        attention_bias = False,
        attention_dropout = 0.0,
        pad_token_id = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig:
    """
    Configuration class for PaliGemma model.
    
    This class holds configuration parameters for both vision and text components
    of the PaliGemma multimodal model, which combines SigLIP vision encoder with
    Gemma language model.
    
    1. vision_config: Configuration dictionary for the SigLIP vision encoder that processes images.
    2. text_config: Configuration dictionary for the Gemma language model that handles text generation.
    3. ignore_index: Token index (-100 by default) to ignore during loss calculation, typically used for padding tokens.
    4. image_token_index: Special token ID (256000) that represents where image content should be inserted in text.
    5. vocab_size: Size of the vocabulary (257152) for the language model, later overwritten by text_config.vocab_size.
    6. projection_dim: Dimension (2048) used for projecting vision features to match language model dimensions.
    7. hidden_size: Base dimension size (2048) for internal representations in the model.
    8. pad_token_id: Token ID used for padding sequences to a consistent length.

    - num_image_tokens: Number of image patch tokens based on image size and patch size
    - is_encoder_decoder: Set to False as PaliGemma is a decoder-only architecture

    """
    def __init__(
        self,
        vision_config = None,
        text_config = None,
        ignore_index = -100,
        image_token_index = 256000,
        vocab_size = 257152,
        projection_dim = 2048,
        hidden_size = 2048,
        pad_token_id = None,
        **kwargs,      
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.vision_config = vision_config
        self.is_encoder_decoder = False

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id = pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    This layer applies RMS normalization to the input tensor.
    RMS normalization is a variant of Layer Normalization that uses
    the root mean square of the activations instead of the mean.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float): A small value added to the denominator for numerical stability.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float()) 
        return output.type_as(x)


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_projection = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
        self.up_projection = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
        self.down_projection = nn.Linear(self.intermediate_size, self.hidden_size, bias = False)

    def forward(self, x):
        # Equivalent to:
        # y = self.gate_projection(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_projection(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_projection(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_projection(nn.functional.gelu(self.gate_projection(x), approximate="tanh") * self.up_projection(x))

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This function is used to repeat the key and value tensors for each head in the attention mechanism because we don't have custom CUDA kernels for the same.
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, slen, n_rep, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim) 

class GemmaRotaryEmbedding(nn.Module):
    """
    Rotary position embeddings for the transformer model.

    This module calculates the rotary position embeddings for the transformer model.
    The embeddings are used to rotate the key and value vectors in the attention mechanism.
    """
    def __init__(self, dim, max_position_embeddings = 2048, base = 10000, device = None):
        super().__init__()
        self.dim = dim # set to head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Calculating the theta according to the formula: theta_i = base ^ (-2i / dim) where i is the head index and dim is the head dimension and i = 0, 1, 2, ..., dim // 2 
        # x^-6 = 1 / x^6
        inverse_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inverse_freq", tensor=inverse_freq, persisitent=False)

    @torch.no_grad()
    def forward(self, x , position_ids, seq_len = None):
        # x:  [Batch_Size,Num_Attention_Heads, Seq_Len, Head_Dim]
        self.inverse_freq.to(x.device)
        # Copying the inverse_freq tensor for batch in the sequence
        # inverse_frequency_expanded: [Batch_Size, Head_Dim //2, 1]
        inverse_frequency_expanded = self.inverse_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None,  :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled= False):
            # Multiplying each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim // 2, 1] @ [Batch_Size, 1, Seq_Len] --> [Batch_Size, Seq_Len, Head_Dim // 2]
            freqs = (inverse_frequency_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # embeddings: [Batch_Size, Seq_Len, Head_Dim]
            embeddings = torch.cat((freqs, freqs), dim = -1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = embeddings.cos()
            sin = embeddings.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x): # implementation is  inspired from hugging face not actual rotary positional encoding paper
    # Build the [-x2, x1, -x4, x3,...] tensor for the sin part of the positional encoding
    x1 = x[..., : x.shape[-1] // 2] # takes the first half of the last dimension of x
    x2 = x[..., x.shape[-1] // 2 :] # takes the second half of the last dimension of x
    return torch.cat((-x2, x1), dim = -1) # concatenates the two halves of x


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
    # q: [Batch_Size, Num_Attention_Heads, Seq_Len, Head_Dim]
    # k: [Batch_Size, Num_Key_Value_Heads, Seq_Len, Head_Dim]
    # cos: [Batch_Size, Seq_Len, Head_Dim]
    # sin: [Batch_Size, Seq_Len, Head_Dim]
    # unsqueeze_dim: 1
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension to the cos and sin tensors
    sin = sin.unsqueeze(unsqueeze_dim)
    # Applying the formula (34) of the Rotary Positional Encoding Paper
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads 
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0, "Hidden size must be divisible by number of heads"
        # assert self.num_key_value_heads % self.num_key_value_groups == 0, "Number of key-value heads must be divisible by number of key-value groups"

        # Number of heads = 8
        # Head_dim = 1024 / 8 = 128
        # Hidden_size = 1024
        # num_key_value_heads = 2
        # num_key_value_groups = 8 / 2 = 4
        # Wq: [ 1024, 8 * 128] = [1024, 1024]
        # Wk: [ 1024, 2 * 128] = [1024, 256]
        # Wv: [ 1024, 2 * 128] = [1024, 256]
        # Wo: [ 128 * 8, 1024] = [1024, 1024]
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias = config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias = config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor] ,Optional[Tuple[ torch.Tensor]]]:

       
        # [Batch_Size, Seq_Len, Hidden_Size]
        batch_size, seq_length, _ = hidden_states.size()

        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)

        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)

        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(hidden_states)

        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states,position_ids, seq_len = None)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim],[Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat the key and values to match the number of heads of the query
        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # Perform the calculation of the attention scores , Query * Key^T / sqrt(Head_Dim) Shape: [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] * [Batch_Size, Num_Heads_KV, Head_Dim, Seq_Len_KV] = [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attention_weights = attention_weights + attention_mask

        # Applying the softmax to the attention weights
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attention_probs = nn.functional.softmax(attention_weights, dim = -1, dtype = torch.float32).to(query_states.dtype)

        # Applying the dropout to the attention probabilities
        attention_probs = nn.functional.dropout(attention_probs, p = self.attention_dropout, training = self.training)

        # Multiplication of the attention probabilities with the values
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] X [Batch_Size, Num_Heads_KV, Seq_Len_KV, Head_Dim] --> [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]
        attention_output = torch.matmul(attention_probs, value_states)

        if attention_output.size() != (batch_size, self.num_heads, seq_length, self.head_dim):
            raise ValueError(
                f"`attention_output` should be of size {(batch_size, self.num_heads, seq_length, self.head_dim)}, but is"
                f" {attention_output.size()}"
            )

        # making sure the sequence length is the second dimension of the output # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] --> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim] 
        attention_output = attention_output.transpose(1, 2).contiguous()

        # Concatenating all the heads together # [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim] --> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        attention_output = attention_output.view(batch_size, seq_length,-1)  # or attention_output.view(batch_size, seq_length, self.num_heads * self.head_dim)

        # Multiplying by W_o # [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim] --> [Batch_Size, Seq_Len_Q, Hidden_Size]
        attention_output = self.o_proj(attention_output)

        return attention_output, attention_probs


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attention = GemmaAttention(config = config, layer_idx = layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
         
        residual = hidden_states # skip connection
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states



class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            num_embeddings = config.vocab_size,
            embedding_dim = config.hidden_size,
            padding_idx = self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def get_input_embeddings(self):  # (input_embeddings ---> text_embeddings + image_embeddings)
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:

        # inputs_embeds: [Batch_size, Seq_len, Hidden_Size]
        # attention_mask: [Batch_size, Seq_len]
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype = hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [Batch_size, Seq_len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states = hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                kv_cache = kv_cache,
            )

        # [Batch_size, Seq_len, Hidden_Size] 
        hidden_states = self.norm(hidden_states)

        return hidden_states

    
    
        

        
class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(in_features = config.hidden_size, out_features = config.vocab_size, bias = False) # layer to convert the hidden states to logits
        

    def tie_weights(self):
        """
        Tie the weights of the language model's output layer with the input embeddings.
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # inputs_embeds: [Batch_size, Seq_len, Hidden_Size]
        # attention_mask: [Batch_size, Seq_len]
        # position_ids: [Batch_size, Seq_len]
        # kv_cache: [Batch_size, Seq_len, Hidden_Size]
        # output: [Batch_size, Seq_len, Hidden_Size]
        outputs = self.model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            inputs_embeds = inputs_embeds,
            kv_cache = kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # returning the updated kv_cache
            return_data["kv_cache"] = kv_cache
        
        return return_data



class PaliGemmaMultiModalProjector(nn.Module):
    """
    This class is responsible for projecting the output of the vision tower to the same dimension as the text embeddings.
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(in_features = config.vision_config.hidden_size, out_features = config.vision_config.projection_dim, bias = True)

    def forward(self, image_features: torch.Tensor):
        # shape: [Batch_size, Num_patches, Embedding_dim]  ---> [Batch_size, Num_patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states




class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # scaling the image features
        scaled_image_features = image_features / (self.config.hidden_size**0.5) # [Batch_Size, Seq_len, Hidden_Size]

        # Combining the embeddings of the text tokens and the image tokens and also masking out all the padding tokens
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, stype= inputs_embeds.dtype, device = inputs_embeds.device)
        # shape: [Batch_Size, Seq_len] ---> True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)

        # shape: [Batch_Size, Seq_len] ---> True for image tokens
        image_mask = (input_ids == self.config.image_token_index)

        # shape: [Batch_Size, Seq_len] ---> True for padding tokens
        padding_mask = (input_ids == self.pad_token_id)

        # We need to expand the masks to the embedding dimension otherwise we  can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        padding_mask_expanded = padding_mask.unsqueeze(-1).expand(-1, -1, embed_dim) 

        # Adding the text embeddings to the final embedding
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)

        # Adding the image embeddings to the final embedding
        # here we can't use torch.where because the sequence length of scaled_image_features is different from the sequence length of final_embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        # Zero out padding tokens
        final_embedding = torch.where(padding_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #=== CREATING THE ATTENTION MASK ===#
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Adding the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids



    # (tokenized_tokens ---> image tokens + bos token + prefix prompt + new line token)  replacing the image tokens with the image embeddings
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

     assert torch.all(attention_mask == 1), "The input cannot contain padding tokens."

     #  Extarcting the input embeddings(image tokens + bos token + prefix prompt + new line token)
     # shape: (batch_size, seq_len, hidden_size)
     inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

     # Merge text and images
     # shape: [Batch_size, Channels, Height, Width] ---> [Batch_size, Num_patches, Embedding_dim]
     selected_image_features = self.vision_tower(pixel_values.to(input_embeds.dtype))

     #  shape: [Batch_size, Num_patches, Embedding_dim]  ---> [Batch_size, Num_patches, Hidden_Size]
     image_features = self.multi_modal_projector(selected_image_features)   # basically making the image embeddings size same as the text embeddings size

     # Merge the embeddings of the text tokens and the image tokens
     inputs_embeds, attention_mask, position_ids = self.merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

     outputs = self.language_model(
         attention_mask = attention_mask,
         inputs_embeds = inputs_embeds,
         position_ids = position_ids,
         kv_cache = kv_cache,
     )

     return outputs
        
        
    
