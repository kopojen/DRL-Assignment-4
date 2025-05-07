# student_agent.py

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


def flatten_observation(time_step):
    """
    將 dm_control 回傳的 observation dict 攤平成一維向量，並回傳 reward 與 done flag。

    Args:
        time_step: dm_control.environment TimeStep 物件

    Returns:
        o_flat (np.ndarray): 攤平後的 observation 向量，dtype=np.float32
        r       (float):     當前 step 的 reward
        done    (bool):      episode 是否結束
    """
    # 以 float32 初始化，避免後續 dtype 不一致
    o_flat = np.array([], dtype=np.float32)
    for key, value in time_step.observation.items():
        arr = np.array(value).astype(np.float32)
        o_flat = np.concatenate((o_flat, arr.flatten()))
    r = time_step.reward
    done = time_step.last()
    return o_flat, r, done

class PolicyNetwork(torch.nn.Module):
    """
    Actor 網路：兩層 256 隱藏層，全連接輸出 mu 和 log_sigma，
    sampling 時用 reparameterization trick，最後經 tanh 約束動作於 [-1,1]。
    """
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mu = torch.nn.Linear(256, action_size)
        self.log_sigma = torch.nn.Linear(256, action_size)

    def forward(self, x: torch.Tensor,
                deterministic: bool = False,
                with_logprob: bool = False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.mu(h2)

        if deterministic:
            return torch.tanh(mu), None

        log_sigma = self.log_sigma(h2).clamp(min=-20.0, max=2.0)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        x_t = dist.rsample()

        if with_logprob:
            log_p = dist.log_prob(x_t).sum(axis=1)
            log_p -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(axis=1)
        else:
            log_p = None

        action = torch.tanh(x_t)
        return action, log_p

class Agent:
    """
    Agent 透過已訓練好的 actor network 給出動作。
    """
    def __init__(self,
                 ckpt_path: str = "400.ckpt",
                 obs_dim: int = 67,
                 act_dim: int = 21):
        # 定義裝置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 建立 actor network → 之後直接轉成 float64 以符合訓練時的 dtype
        self.actor = PolicyNetwork(obs_dim, act_dim).to(self.device).double()
        # 載入 checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])  # checkpoint 亦為 float64 權重
        self.actor.eval()

    def act(self, obs_input) -> np.ndarray:
        """輸入 Numpy 觀測或 dm_control TimeStep，輸出 [-1,1] 動作 (np.float64)。"""
        if isinstance(obs_input, np.ndarray):
            obs = obs_input.astype(np.float64)
        else:
            obs, _, _ = flatten_observation(obs_input)
            obs = obs.astype(np.float64)

        obs_t = torch.as_tensor(obs, dtype=torch.float64, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, _ = self.actor(obs_t, deterministic=True)
        return action_t.cpu().numpy().flatten().astype(np.float64)