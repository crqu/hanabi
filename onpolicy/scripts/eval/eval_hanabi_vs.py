#!/usr/bin/env python
import sys
import os
from pathlib import Path
import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.envs.hanabi.Hanabi_Env import HanabiEnv
from onpolicy.envs.env_wrappers import ChooseDummyVecEnv


def build_env(all_args):
    """Single-thread env for evaluation."""
    def env_fn():
        assert 1 < all_args.num_agents < 6, "num_agents must be 2-5 for Hanabi"
        env = HanabiEnv(all_args, all_args.seed)
        env.seed(all_args.seed)
        return env
    return ChooseDummyVecEnv([env_fn])


def load_policy(all_args, model_dir, device):
    """Construct policy and load actor weights from model_dir."""
    policy = R_MAPPOPolicy(
        all_args,
        obs_space=None,  # filled after env creation
        cent_obs_space=None,
        act_space=None,
        device=device,
    )
    actor_path = os.path.join(model_dir, "actor.pt")
    state_dict = torch.load(actor_path, map_location=device)
    policy.actor.load_state_dict(state_dict)
    return policy


def evaluate_vs(all_args):
    # Force IPPO-style critic usage for vs mode.
    all_args.use_centralized_V = False
    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")

    envs = build_env(all_args)
    obs_space = envs.observation_space[0]
    act_space = envs.action_space[0]

    # Build two policies (actors) and load weights.
    pol_a = R_MAPPOPolicy(all_args, obs_space, obs_space, act_space, device=device)
    pol_b = R_MAPPOPolicy(all_args, obs_space, obs_space, act_space, device=device)
    pol_a.actor.load_state_dict(torch.load(os.path.join(all_args.model_dir_a, "actor.pt"), map_location=device))
    pol_b.actor.load_state_dict(torch.load(os.path.join(all_args.model_dir_b, "actor.pt"), map_location=device))

    # RNN states per agent
    rnn_a = np.zeros((1, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    rnn_b = np.zeros_like(rnn_a)

    scores = []
    episodes = all_args.eval_episodes
    for _ in range(episodes):
        obs, share_obs, available_actions = envs.reset(np.ones(1, dtype=bool))
        # reset RNN/masks each episode
        rnn_a.fill(0)
        rnn_b.fill(0)
        masks = np.ones((1, 1), dtype=np.float32)
        done = False

        while not done:
            env_actions = np.ones((1, all_args.num_agents, 1), dtype=np.float32) * -1
            for aid in range(all_args.num_agents):
                if not available_actions[0, aid].any():
                    continue
                obs_in = obs[:, aid]  # shape (1, obs_dim)
                rnn_in = rnn_a if aid == 0 else rnn_b
                # actor.forward expects masks shape (batch, 1)
                action, _, rnn_out = (pol_a.actor if aid == 0 else pol_b.actor)(
                    obs_in, rnn_in, masks, available_actions[:, aid], deterministic=all_args.deterministic_eval
                )
                env_actions[0, aid] = action.cpu().numpy()
                if aid == 0:
                    rnn_a = rnn_out.cpu().numpy()
                else:
                    rnn_b = rnn_out.cpu().numpy()

            obs, share_obs, rewards, dones, infos, available_actions = envs.step(env_actions)
            done = bool(dones[0])
            if done and infos[0].get("score") is not None:
                scores.append(infos[0]["score"])

    envs.close()
    avg_score = float(np.mean(scores)) if scores else 0.0
    print(f"Evaluated {episodes} episodes. Avg score: {avg_score:.3f}")
    return avg_score, scores


def parse_args():
    parser = get_config()
    parser.add_argument("--model_dir_a", type=str, required=True, help="Path to first agent's model directory (expects actor.pt)")
    parser.add_argument("--model_dir_b", type=str, required=True, help="Path to second agent's model directory (expects actor.pt)")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--deterministic_eval", action="store_true", default=False, help="Use greedy actions during evaluation")
    # Defaults that make sense for vs eval
    parser.set_defaults(use_eval=True, share_policy=False, n_rollout_threads=1, n_eval_rollout_threads=1, use_wandb=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    evaluate_vs(args)


if __name__ == "__main__":
    main()
