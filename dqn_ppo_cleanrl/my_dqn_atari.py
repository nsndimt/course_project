# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import datetime
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--capture-video", action='store_true',
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", action='store_true',
        help="whether to save model into the `runs/{run_name}` folder")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=40,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=512,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=8000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=1,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on

    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env

    return thunk


class NoisyLinear(nn.Module):
    """Implementation of Noisy Networks. arXiv:1706.10295.

    :param int in_features: the number of input features.
    :param int out_features: the number of output features.
    :param float noisy_std: initial standard deviation of noisy linear layers.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, in_features: int, out_features: int, noisy_std: float = 0.5) -> None:
        super().__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer("eps_W", torch.FloatTensor(out_features, in_features))
        self.register_buffer("eps_bias", torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = noisy_std

        self.reset()
        self.sample()

    def reset(self) -> None:
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.in_features))
    
    def sample(self) -> None:
        self.eps_W.copy_(torch.randn_like(self.eps_W))  # type: ignore
        self.eps_bias.copy_(torch.randn_like(self.eps_bias))  # type: ignore

    def zero(self) -> None:
        self.eps_W.copy_(torch.zeros_like(self.eps_W))  # type: ignore
        self.eps_bias.copy_(torch.zeros_like(self.eps_bias))  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_W + self.sigma_W * self.eps_W # type: ignore
            bias = self.mu_bias + self.sigma_bias * self.eps_bias  # type: ignore
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)


def sample_noise(model: nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, NoisyLinear):
            m.sample()


def disable_noise(model: nn.Module) -> bool:
    for m in model.modules():
        if isinstance(m, NoisyLinear):
            m.zero()


# ALGO LOGIC: initialize agent here:
class DuelingQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.val = nn.Sequential(
            NoisyLinear(3136, 512),
            nn.ReLU(),
            NoisyLinear(512, 1),
        )
        self.adv = nn.Sequential(
            NoisyLinear(3136, 512),
            nn.ReLU(),
            NoisyLinear(512, env.single_action_space.n),
        )

    def forward(self, x):
        x = self.network(x / 255.0)
        val = self.val(x)
        adv = self.adv(x)
        adv_ave = torch.mean(adv, dim=1, keepdim=True)
        return adv + val - adv_ave


if __name__ == "__main__":
    args = parse_args()
    now = datetime.datetime.now()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{now.isoformat()}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, False, run_name) for i in range(args.num_envs)], context="spawn"
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = DuelingQNetwork(envs).to(device)
    q_network.train()
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = DuelingQNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.train()

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here

        with torch.inference_mode():
            sample_noise(q_network)
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        if (global_step + 1) % 1000 == 0:
            end_time = time.time()
            # print("SPS:", int(1000 / (end_time - start_time))
            writer.add_scalar("charts/SPS", int(1000 / (end_time - start_time)) * args.num_envs, global_step + 1)
            start_time = end_time

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None or "episode" not in info:
                    continue
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step + 1)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step + 1)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = [infos["final_observation"][idx] if t else next_obs[idx] for idx, t in enumerate(truncated)]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step + 1 > args.learning_starts:
            if (global_step + 1) % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                sample_noise(q_network)
                best_next_action = torch.argmax(q_network(data.next_observations), 1, keepdim=True)
                with torch.no_grad():
                    sample_noise(target_network)
                    target_max = target_network(data.next_observations).gather(1, best_next_action).squeeze(1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze(1)
                loss = F.mse_loss(td_target, old_val)

                if (((global_step + 1) // args.train_frequency) + 1) % 10 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step + 1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step + 1)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, 0, args.capture_video, run_name)])
    obs, _ = envs.reset()
    q_network.eval()
    disable_noise(q_network)

    episodic_returns = []
    eval_episodes = 20
    while len(episodic_returns) < eval_episodes:
        with torch.inference_mode():
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns.append(info["episode"]["r"])
        obs = next_obs

    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
