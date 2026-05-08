import torch
from torch import nn

class CustomResNetEmbedding(nn.Module):
    def __init__(self, embedder, output_size):
        super().__init__()
        self.embedder = embedder
        conv = embedder.embedder.convolution
        self.conv = conv
        w = conv.weight
        w_flat = w.view(w.shape[0], -1)
        self.weight = w_flat
        self.output_size = output_size
        
    def forward(self, pixel_values):
        embedding = self.weight.to(pixel_values.device) @ pixel_values
        embedding = embedding.view(pixel_values.shape[0], *self.output_size[1:])
        embedding = self.embedder.embedder.normalization(embedding)
        embedding = self.embedder.embedder.activation(embedding)
        embedding = self.embedder.pooler(embedding)
        return embedding

class CustomPatchEmbed(nn.Module):
    def __init__(self, patch_embed, output_size=None, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.convolution = patch_embed.conv_down[0]
        w = self.convolution.weight
        w_flat = w.view(w.shape[0], -1)
        self.register_buffer("weight", w_flat)

        self.conv_down = nn.Sequential(
            patch_embed.conv_down[1],
            patch_embed.conv_down[2],
            patch_embed.conv_down[3],
            patch_embed.conv_down[4],
            patch_embed.conv_down[5],
        )
        self.patch_embed = patch_embed
        self.output_size=output_size

    def forward(self, x):
        x = self.proj(x)
        x = self.weight @ x
        x = x.view(x.shape[0], *self.output_size[1:])
        x = self.conv_down(x)
        return x

