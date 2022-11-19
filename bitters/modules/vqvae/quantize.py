import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

class VectorQuantizer(nn.Module):
    def __init__(self, num_tokens, codebook_dim, beta=0.25):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(self.num_tokens, self.codebook_dim)
        self.beta = beta

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
                        
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c')
        z_flattened = l2norm(z.reshape(-1, self.codebook_dim))
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            l2norm(self.embedding.weight).pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, l2norm(self.embedding.weight)) # 'n d -> d n'

        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z) + F.mse_loss(z_q, z.detach())

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, 'b h w c -> b c h w')

        return loss, z_q, encoding_indices
