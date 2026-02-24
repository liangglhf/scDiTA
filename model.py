import torch
import torch.nn as nn
from typing import Optional
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNormZero

class DiT(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        input_dim: int = 16,            # Dimension after PCA
        num_layers: int = 16,           # Depth: 16
        hidden_dim: int = 384,          # Hidden layer: 384
        num_heads: int = 6,             # Muti-head: 6
        seq_len: int = 64,              # gene cluster 
        num_classes: int = 64,          # cell type(after PCA)  
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        # input data after PCA (No Patchify)
        self.x_embedder = nn.Linear(input_dim, hidden_dim)
        # Positional Encoding (1, 64, 384) 
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        # Conditional Embedding (Time + Class)
        # timestep
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_dim)
        # class, CFG
        self.class_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        # AdaLN Modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        # Transformer Backbone
        self.blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=hidden_dim,
                num_attention_heads=num_heads,
                attention_head_dim=hidden_dim // num_heads,
                dropout=0.0,
                cross_attention_dim=None,
                activation_fn="gelu-approximate",
                attention_bias=False,
                only_cross_attention=False,
                norm_type="ada_norm_single"
            )
            for _ in range(num_layers)
        ])
        # Final layer
        self.final_norm = AdaLayerNormZero(embedding_dim=hidden_dim, num_embeddings=None)
        self.final_proj = nn.Linear(hidden_dim, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)
        nn.init.constant_(self.final_proj.weight, 0)
        nn.init.constant_(self.final_proj.bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        return_intermediate_layers: bool = False,
        target_layer_idx: int = 8
    ):
        # Embedding Inputs
        x = self.x_embedder(sample) + self.pos_embed # (B, 64, 384) 
        # Conditions
        # timestep
        t_emb = self.time_proj(timestep)
        t_emb = self.time_embedding(t_emb) # (B, 384)
        # class
        if class_labels is None:
            # default Null Token 
            class_labels = torch.full((sample.shape[0],), self.num_classes, device=sample.device).long()
        y_emb = self.class_embedding(class_labels) # (B, 384)
        c_raw = t_emb + y_emb 
        c_for_blocks = self.adaLN_modulation(c_raw)
        # Transformer Blocks Forward
        intermediate_output = None
        for i, block in enumerate(self.blocks):
            x = block(x, timestep=c_for_blocks) 
            if return_intermediate_layers and i == target_layer_idx:
                intermediate_output = x.clone()
        # Velocity Prediction
        x = self.final_norm(x, emb=c_raw)[0]
        x = self.final_proj(x) # (B, 64, 16)
        if return_intermediate_layers:
            return x, intermediate_output
        return x