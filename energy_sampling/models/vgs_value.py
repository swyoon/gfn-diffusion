import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = timesteps.type(dtype=torch.float)[:, None] * emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.pad(emb, [0, 1], value=0.0)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def process_single_t(x, t):
    """make single integer t into a vector of an appropriate size"""
    if isinstance(t, float):
        t = torch.ones([x.shape[0]], dtype=torch.float, device=x.device) * t
    if isinstance(t, int) or len(t.shape) == 0 or len(t) == 1:
        t = torch.ones([x.shape[0]], dtype=torch.long, device=x.device) * t
    return t



class FCNet_temb_deeperv3(nn.Module):
    """
    New FCNet_temb_deeper architecture inspired by models/wideresnet_te/wideresnet_te.py from GCD repo
    """
    def __init__(
        self, in_dim, out_dim, hidden_dim=1024, t_emb_dim=128, t_feature_dim = 512,
    ):
        super().__init__()
        # fc layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)

        # temb layers
        self.temb = nn.Linear(t_emb_dim, t_feature_dim)
        self.temb1 = nn.Linear(t_feature_dim, hidden_dim)
        self.temb2 = nn.Linear(t_feature_dim, hidden_dim)

        self.t_emb_dim = t_emb_dim


    def forward(self, x, t):
        """x is (batch, feature_dim)"""
        # get timestep embedding
        t = process_single_t(x, t)
        t_emb = get_timestep_embedding(t, self.t_emb_dim).to(x.device)
        t_emb = self.temb(t_emb)

        x_ = self.fc1(x)
        x_ = torch.cat((x_, self.temb1(t_emb)), dim = 1)
        x_ = F.relu(x_)
        x_ = self.fc2(x_)
        x_ = F.relu(x_)
        x_ = self.fc3(x_)
        x_ *= self.temb2(t_emb)
        x_ = self.fc4(x_)

        return x_



class invariant_wrapper(nn.Module):

    def __init__(self, n_particles, n_dim, net, dis_reciprocal=False, eps = 1.0, zero_init=True):
        super().__init__()
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.net = net # input_dim should be n_particles * (n_particles-1) // 2
        self.dis_reciprocal = dis_reciprocal
        self.eps = eps
        idx = torch.triu_indices(n_particles, n_particles, offset=1)
        self.idx_i = idx[0]
        self.idx_j = idx[1]

        if zero_init:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.fill_(0.0)
                    m.bias.data.fill_(0.01)

    def forward(self, x, t):
        """
        x: torch.Tensor of shape [Batch,  n_particles * n_dimensions]
        t: int
        """
        x = x.view(x.shape[0], self.n_particles, self.n_dim)
        xi = x[:, self.idx_i, :]     # [B, n_particles * (n_particles-1) // 2, D]
        xj = x[:, self.idx_j, :]     # [B, n_particles * (n_particles-1) // 2, D]
        sq_distances = torch.sum((xi - xj) ** 2, dim=-1)  # [B, n_particles * (n_particles-1) // 2]
        sq_distances, _ = torch.sort(sq_distances, dim=-1, descending=True)
        if self.dis_reciprocal:
            input = 1 / (sq_distances + self.eps).sqrt() # To avoid division by zero
        else:
            input = (sq_distances + self.eps).sqrt() # to avoid near-zero distances
        return self.net(input, t)