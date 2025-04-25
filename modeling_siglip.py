from typing import List, Tuple, Optional    
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072, 
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 num_channels: int = 3,
                 image_size: int = 224,
                 patch_size: int = 16,
                 layer_norm_eps: float = 1e-6,
                 attention_dropout: float = 0.0,
                 num_image_tokens: int = None,
                 **kwargs
                 ):

                 super().__init__()

                 self.hidden_size = hidden_size
                 self.intermediate_size = intermediate_size
                 self.num_hidden_layers = num_hidden_layers
                 self.num_attention_heads = num_attention_heads
                 self.num_channels = num_channels
                 self.image_size = image_size
                 self.patch_size = patch_size
                 self.layer_norm_eps = layer_norm_eps
                 self.attention_dropout = attention_dropout
                 self.num_image_tokens = num_image_tokens
                 self.kwargs = kwargs


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embeddings = nn.Conv2d(in_channels = config.num_channels,
                                          out_channels = config.hidden_size,
                                          kernel_size = config.patch_size,
                                          stride = config.patch_size,
                                          padding = "valid")  #  this indicates no padding

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.embedding_dim)  # learning positional embeddings
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent = False)  

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim] # [batch_size, seq_len, embed_dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SigLipMLP(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.hidden_size, self.config.intermediate_size)
        self.fc2 = nn.Linear(self.config.intermediate_size, self.config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate = "tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class SigLipAttention(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads # in Siglip, num_attention_heads = 12
        self.attention_head_size = self.embedding_dim // self.num_attention_heads
        self.scale = self.attention_head_size ** -0.5 # equivalent to 1 / sqrt(attention_head_size)

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        k_proj = self.k_proj(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        v_proj = self.v_proj(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        q_proj = self.q_proj(hidden_states)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Attention_Heads, Num_Patches, Attention_Head_Size]
        k_proj = k_proj.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Attention_Heads, Num_Patches, Attention_Head_Size]
        v_proj = v_proj.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Attention_Heads, Num_Patches, Attention_Head_Size]
        q_proj = q_proj.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Calculating the attention score using the formula : Q * k^T/ sqrt(d_k) where Q, k are the query and key vectors
        attention_weights = torch.matmul(q_proj, k_proj.transpose(2, 3)) * self.scale # [Batch_Size, Num_Attention_Heads, Num_Patches, Num_Patches]

        if attention_weights.size() != (batch_size, self.num_attention_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_attention_heads, seq_len, seq_len)}, but is"
                f" {attention_weights.size()}"
            )

        # Applying the softmax row-wise
        # [Batch_Size, Num_Attention_Heads, Num_Patches, Num_Patches]
        attention_weights = nn.functional.softmax(attention_weights, dim = -1, dtype = torch.float32).to(q_proj.dtype)

        # Applying dropout only during training
        attention_weights = nn.functional.dropout(attention_weights, p = self.config.attention_dropout, training = self.training)

        # Multiplying the attention weights with the value vectors
        attention_output = torch.matmul(attention_weights, v_proj) # [Batch_Size, Num_Attention_Heads, Num_Patches, Attention_Head_Size]

        if attention_output.size() != (batch_size, self.num_attention_heads, seq_len, self.attention_head_size):
            raise ValueError(
                f"`attention_output` should be of size {(batch_size, self.num_attention_heads, seq_len, self.attention_head_size)}, but is"
                f" {attention_output.size()}"
            )

        # [Batch_Size, Num_Attention_Heads, Num_Patches, Attention_Head_Size] -> [Batch_Size, Num_Patches,  Num_Attention_Heads, Attention_Head_Size]
        attention_output = attention_output.transpose(1, 2).contiguous()  # contiguous() ensures that the tensor is stored in a contiguous chunk of memory

        # [Batch_Size, Num_Patches,  Num_Attention_Heads, Attention_Head_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        attention_output = attention_output.reshape(batch_size, seq_len, self.embedding_dim)

        # Multiplying the attention output with the output projection matrix
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attention_output = self.out_proj(attention_output)

        return attention_output, attention_weights
       

class SiglipEncoderLayer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.self_attention = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embedding_dim, eps = config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dim, eps = config.layer_norm_eps)
           

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual connection : [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states  
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states) 
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attention(hidden_states = hidden_states)
        # residual connection : [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual connection : [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states = hidden_states)
        # residual connection : [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states

        return hidden_states

class SiglipVisionEncoder(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.layer = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # inputs_embeds: [Batch_size, Num_patches, Embedding_dim(hidden_size)]
        hidden_states = inputs_embeds
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states = hidden_states)

        return hidden_states

                    
class SiglipVisionTransformer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embedding_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)

        self.post_layernorm = nn.LayerNorm(embedding_dim, eps = config.layer_norm_eps)


    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_size, Channels, Height, Width] -> [Batch_size, Num_patches, Embedding_dim(hidden_size)]
        hidden_states = self.embeddings(pixel_values = pixel_values)

        last_hidden_state = self.encoder(inputs_embeds = hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_patches, Embedding_dim(hidden_size)]
        return self.vision_model(pixel_values = pixel_values)