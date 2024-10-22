import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class AdaptiveWeightNetwork(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, t):
        # [batch_size] -> [batch_size, 2]
        t_emb = torch.stack([
            torch.sin(t * 2.0 * np.pi * 0.02),
            torch.cos(t * 2.0 * np.pi * 0.02)
        ], dim=-1)
        return self.net(t_emb)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        
    def forward(self, x):
        # calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # normalize and scale
        return (x / rms) * self.scale

class AdaptiveDoubleNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.norm = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2)
        )
    
    def forward(self, x, t_emb):
        normalized = self.norm(x)
        scale, bias = self.mlp(t_emb).chunk(2, dim=-1)
        scale = scale / torch.sqrt(torch.mean(scale**2, dim=-1, keepdim=True) + 1e-8)
        bias = bias / torch.sqrt(torch.mean(bias**2, dim=-1, keepdim=True) + 1e-8)
        return normalized * (1 + scale) + bias

class TrigFlowNet(nn.Module):
    def __init__(self, data_dim, hidden_dim=128):
        super().__init__()
        self.data_dim = data_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm_layers = nn.ModuleList([
            AdaptiveDoubleNorm(hidden_dim) for _ in range(2)
        ])
        
        self.main_layers = nn.ModuleList([
            nn.Linear(data_dim + hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        
        self.final = nn.Linear(hidden_dim, data_dim)
    
    def forward(self, x, t):
        batch_size = x.shape[0]
        
        # Create time embeddings [batch_size, 2]
        t_emb = torch.stack([
            torch.sin(t * 2.0 * np.pi * 0.02),
            torch.cos(t * 2.0 * np.pi * 0.02)
        ], dim=-1)
        
        # Process time embedding [batch_size, hidden_dim]
        t_emb = self.time_embed(t_emb)
        
        # Concatenate features [batch_size, data_dim + hidden_dim]
        h = torch.cat([x, t_emb], dim=-1)
        
        for layer, norm in zip(self.main_layers, self.norm_layers):
            h = layer(h)
            h = norm(h, t_emb)
            h = nn.SiLU()(h)
        
        return self.final(h)

class TrigFlow:
    def __init__(self, data_dim, sigma_d=1.0, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = TrigFlowNet(data_dim).to(device)
        self.weight_net = AdaptiveWeightNetwork().to(device)
        
        # self.optimizer = torch.optim.Adam(
        #     [
        #         {'params': self.model.parameters()},
        #         {'params': self.weight_net.parameters(), 'lr': 1e-4}
        #     ],
        #     lr=1e-3, eps=1e-8
        # )
        
        self.optimizer = torch.optim.Adam(
            [
                {'params': self.model.parameters()},
                {'params': self.weight_net.parameters(), 'lr': 1e-3}
            ],
            lr=1e-3, eps=1e-8
        )

        self.sigma_d = sigma_d
        self.data_dim = data_dim
        self.warmup_steps = 10000
        self.current_step = 0
        
    def get_reference_noise(self, x0, t):
        z = torch.randn_like(x0) * self.sigma_d
        return z
    
    def diffuse(self, x0, t):
        z = self.get_reference_noise(x0, t)
        xt = torch.cos(t).unsqueeze(-1) * x0 + torch.sin(t).unsqueeze(-1) * z
        return xt, z
    
    def compute_tangent(self, x, t):
        cos_t = torch.cos(t).unsqueeze(-1)
        sin_t = torch.sin(t).unsqueeze(-1)
        
        pred = self.model(x/self.sigma_d, t) * self.sigma_d
        tangent = -cos_t * pred - sin_t * x
        return tangent
    
    def normalize_tangent(self, tangent, eps=0.1):
        norm = torch.norm(tangent, dim=-1, keepdim=True)
        return tangent / (norm + eps)
    
    def train_step(self, x0):
        x0 = x0.to(self.device)
        batch_size = x0.shape[0]
        
        # Sample time steps
        tau = torch.randn(batch_size, device=self.device) * 0.7 - 1.0
        t = torch.arctan(torch.exp(tau) / self.sigma_d)
        
        # Forward process
        xt, z = self.diffuse(x0, t)
        
        # Compute and normalize tangent
        tangent = self.compute_tangent(xt, t)
        norm_tangent = self.normalize_tangent(tangent)
        
        # Apply warmup
        r = min(1.0, self.current_step / self.warmup_steps)
        norm_tangent = norm_tangent * r
        
        # Compute loss with adaptive weighting
        w = self.weight_net(t)
        prior_weight = 1 / (self.sigma_d * torch.tan(t))
        prior_weight = prior_weight.unsqueeze(-1)
        
        loss = (
            torch.exp(w) / self.data_dim * prior_weight * (norm_tangent ** 2).sum(-1)
        ).mean() - w.mean()
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.current_step += 1
        return loss.item()
    
    def sample(self, batch_size, steps=100, intermediate_time=1.1):
        device = next(self.model.parameters()).device
        x = torch.randn(batch_size, self.data_dim, device=device) * self.sigma_d
        t_max = torch.arctan(torch.tensor(80.0, device=device) / self.sigma_d)
        
        with torch.no_grad():
            if steps == 1:
                v = self.model(x/self.sigma_d, t_max*torch.ones(batch_size, device=device)) * self.sigma_d
                x = torch.cos(t_max) * x - torch.sin(t_max) * v
            else:
                # First step
                t_mid = torch.tensor(intermediate_time, device=device)
                t_batch = t_max * torch.ones(batch_size, device=device)
                v = self.model(x/self.sigma_d, t_batch) * self.sigma_d
                x = torch.cos(t_max - t_mid) * x - torch.sin(t_max - t_mid) * v
                
                # Second step
                t_batch = t_mid * torch.ones(batch_size, device=device)
                v = self.model(x/self.sigma_d, t_batch) * self.sigma_d
                x = torch.cos(t_mid) * x - torch.sin(t_mid) * v
        
        return x

def generate_circle_data(n_samples, sigma=0.1, device="cuda" if torch.cuda.is_available() else "cpu"):
    theta = torch.rand(n_samples, device=device) * 2 * np.pi
    r = torch.ones(n_samples, device=device) + torch.randn(n_samples, device=device) * sigma
    x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    return x

def train_and_visualize():
    # 训练参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    n_samples = 1000
    batch_size = 100
    n_epochs = 5000
    data_dim = 2
    
    # 初始化模型
    flow = TrigFlow(data_dim, device=device)
    
    # 训练循环
    losses = []
    progress_bar = tqdm(range(n_epochs))
    for epoch in progress_bar:
        x0 = generate_circle_data(batch_size, device=device)
        loss = flow.train_step(x0)
        losses.append(loss)
        
        if (epoch + 1) % 100 == 0:
            progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # 生成样本并可视化
    samples = flow.sample(n_samples).cpu().numpy()
    real_data = generate_circle_data(n_samples, device=device).cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    
    # Plot real data
    plt.subplot(131)
    plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, label='Real')
    plt.title("Real Data")
    plt.legend()
    
    # Plot generated samples
    plt.subplot(132)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='Generated')
    plt.title("Generated Samples")
    plt.legend()
    plt.savefig('gen.png')
    
    # Plot loss curve
    plt.subplot(133)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig('trigflow_results.png')
    plt.show()

if __name__ == "__main__":
    train_and_visualize()
