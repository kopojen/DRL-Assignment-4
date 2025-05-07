# student_agent.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

class PolicyNetwork(torch.nn.Module):
    """
    Actor 網路：兩層 256 隱藏層，全連接輸出 mu 和 log_sigma，
    取樣時用 reparameterization trick，最後經 tanh 約束動作於 [-1,1]。
    """
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mu = torch.nn.Linear(256, action_size)
        self.log_sigma = torch.nn.Linear(256, action_size)

    def forward(self,
                x: torch.Tensor,
                deterministic: bool = False,
                with_logprob: bool = False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.mu(h2)

        if deterministic:
            return torch.tanh(mu), None

        log_sigma = self.log_sigma(h2).clamp(min=-20.0, max=2.0)
        sigma     = torch.exp(log_sigma)
        dist      = Normal(mu, sigma)
        x_t       = dist.rsample()

        if with_logprob:
            log_p = dist.log_prob(x_t).sum(axis=1)
            # tanh 校正項
            log_p -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(axis=1)
        else:
            log_p = None

        action = torch.tanh(x_t)
        return action, log_p

class Agent:
    """
    只支援助教的 make_dmc_env env──
    接收已展平成一維 numpy array 的 observation，
    輸出 shape=(action_dim,) 的 np.float64 動作向量。
    """
    def __init__(self,
                 ckpt_path: str = "500.ckpt",
                 obs_dim:   int = 67,
                 act_dim:   int = 21):
        # 設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 建 actor network 並轉成 float64
        self.actor = PolicyNetwork(obs_dim, act_dim).to(self.device).double()
        # 載入權重
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        只接受已展平成一維 float64 numpy array 的 obs，
        回傳 float64 numpy array，shape=(act_dim,)。
        """
        if not isinstance(obs, np.ndarray):
            raise ValueError("觀測必須是 numpy array")
        obs = obs.astype(np.float64)
        if obs.ndim != 1:
            raise ValueError(f"觀測向量必須是一維，got {obs.shape}")

        # 推論
        obs_t = torch.as_tensor(obs, dtype=torch.float64, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, _ = self.actor(obs_t, deterministic=True)
        return action_t.cpu().numpy().flatten().astype(np.float64)
