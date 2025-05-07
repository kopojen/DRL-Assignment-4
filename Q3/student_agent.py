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
        o_flat (np.ndarray): 攤平後的 observation 向量
        r       (float):     當前 step 的 reward
        done    (bool):      episode 是否結束
    """
    o_flat = np.array([])
    for key, value in time_step.observation.items():
        if value.shape:  # array
            o_flat = np.concatenate((o_flat, value.flatten()))
        else:            # scalar
            o_flat = np.concatenate((o_flat, np.array([value])))
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
        self.mu_layer        = torch.nn.Linear(256, action_size)
        self.log_sigma_layer = torch.nn.Linear(256, action_size)

    def forward(self,
                x: torch.Tensor,
                deterministic: bool = False,
                with_logprob: bool = False):
        """
        Args:
            x            (Tensor):   shape=(B, obs_size)
            deterministic (bool):    是否只回傳 tanh(mu)；True 用於 eval
            with_logprob (bool):     是否計算 log_prob；用於 train

        Returns:
            action   (Tensor): shape=(B, action_size)，經 tanh 後的動作
            log_prob (Tensor|None): 若 with_logprob，回傳加了 tanh 修正的 log_prob
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.mu_layer(h2)

        if deterministic:
            return torch.tanh(mu), None

        log_sigma = self.log_sigma_layer(h2).clamp(min=-20.0, max=2.0)
        sigma     = torch.exp(log_sigma)
        dist      = Normal(mu, sigma)
        x_t       = dist.rsample()

        if with_logprob:
            log_p = dist.log_prob(x_t).sum(axis=1)
            log_p -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(axis=1)
        else:
            log_p = None

        action = torch.tanh(x_t)
        return action, log_p

class Agent:
    def __init__(self, 
                 obs_dim: int = 67,
                 act_dim: int = 21):
        ckpt_path = "350.ckpt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor  = PolicyNetwork(obs_dim, act_dim).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()

    def act(self, time_step) -> np.ndarray:
        """
        根據 dm_control 回傳的 time_step，輸出一維動作向量。

        Args:
            time_step: dm_control.environment TimeStep 物件

        Returns:
            action (np.ndarray): shape=(act_dim,) 且每個元素 ∈ [-1,1]
        """
        obs, _, _ = flatten_observation(time_step)
        obs_t = torch.as_tensor(obs, dtype=torch.float64,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, _ = self.actor(obs_t, deterministic=True)
        return action_t.cpu().numpy().flatten()