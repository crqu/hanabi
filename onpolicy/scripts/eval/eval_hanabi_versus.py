#!/usr/bin/env python
import sys
import os
import numpy as np
import copy
from pathlib import Path
import torch

from onpolicy.config import get_config
from onpolicy.envs.hanabi.Hanabi_Env import HanabiEnv
from onpolicy.envs.env_wrappers import ChooseSubprocVecEnv, ChooseDummyVecEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Hanabi":
                assert 1 < all_args.num_agents < 6, "num_agents must be 2-5."
                env = HanabiEnv(all_args, (all_args.seed * 50000 + rank * 10000))
            else:
                raise NotImplementedError(f"Unsupported env {all_args.env_name}")
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--hanabi_name", type=str, default="Hanabi-Full", help="Which Hanabi env to run on")
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")
    parser.add_argument("--actor_path_a", type=str, required=True, help="Path to agent0 actor checkpoint")
    parser.add_argument("--actor_path_b", type=str, required=True, help="Path to agent1 actor checkpoint")
    parser.add_argument("--use_recurrent_a", action="store_true", default=False, help="Whether agent0 actor uses RNN")
    parser.add_argument("--use_recurrent_b", action="store_true", default=False, help="Whether agent1 actor uses RNN")
    parser.add_argument("--deterministic_eval", action="store_true", default=False, help="Use greedy actions during eval")
    parser.add_argument("--log_actions_path", type=str, default=None, help="If set, save per-step actions for the first episode/env to this file (jsonl).")
    return parser.parse_known_args(args)[0]


def load_actor(all_args, envs, actor_path, device, use_recurrent):
    """
    Load only the actor network for evaluation.
    """
    tmp_args = copy.copy(all_args)
    tmp_args.use_recurrent_policy = use_recurrent
    tmp_args.use_naive_recurrent_policy = use_recurrent
    share_observation_space = envs.share_observation_space[0] if tmp_args.use_centralized_V else envs.observation_space[0]
    policy = Policy(tmp_args,
                    envs.observation_space[0],
                    share_observation_space,
                    envs.action_space[0],
                    device=device)
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"Actor path does not exist: {actor_path}")
    policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
    policy.actor.eval()
    return policy.actor


@torch.no_grad()
def evaluate(all_args):
    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    envs = make_eval_env(all_args)
    num_agents = all_args.num_agents

    actor_paths = [all_args.actor_path_a, all_args.actor_path_b]
    recur_flags = [all_args.use_recurrent_a, all_args.use_recurrent_b]
    actors = []
    for aid in range(num_agents):
        ap = actor_paths[aid] if aid < len(actor_paths) else actor_paths[-1]
        use_recur = recur_flags[aid] if aid < len(recur_flags) else recur_flags[-1]
        actors.append(load_actor(all_args, envs, ap, device, use_recur))

    total_rewards = []
    action_log = [] if all_args.log_actions_path else None
    for ep in range(all_args.eval_episodes):
        # Hanabi wrapper expects reset_choose mask; start with all True
        reset_choose = np.ones(all_args.n_eval_rollout_threads, dtype=bool)
        obs, share_obs, available_actions = envs.reset(reset_choose)
        share_obs = share_obs if all_args.use_centralized_V else obs

        rnn_states = np.zeros((all_args.n_eval_rollout_threads, num_agents, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        masks = np.ones((all_args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
        ep_rew = np.zeros((all_args.n_eval_rollout_threads, num_agents), dtype=np.float32)

        for _ in range(all_args.episode_length):
            actions_env = np.zeros((all_args.n_eval_rollout_threads, num_agents), dtype=np.int64)
            for aid in range(num_agents):
                agent_share_obs = share_obs[:, aid] if share_obs.ndim >= 3 else share_obs
                agent_obs = obs[:, aid] if obs.ndim >= 3 else obs
                agent_avail = available_actions[:, aid] if available_actions is not None and available_actions.ndim >= 3 else available_actions

                action, action_log_probs, rnn_state = actors[aid](
                    agent_obs,
                    rnn_states[:, aid],
                    masks[:, aid],
                    agent_avail,
                    deterministic=all_args.deterministic_eval
                )
                act_np = action.detach().cpu().numpy()
                if act_np.ndim > 1:
                    act_np = act_np.squeeze(-1)
                actions_env[:, aid] = act_np
                rnn_states[:, aid] = rnn_state.cpu().numpy()

            obs, share_obs, rewards, dones, infos, available_actions = envs.step(actions_env)
            rewards = np.asarray(rewards)
            # Hanabi wrapper may return rewards with an extra agent axis; collapse to [threads, agents]
            if rewards.ndim > 2:
                rewards = rewards.sum(axis=-1)
            if rewards.ndim == 3 and rewards.shape[-1] == 1:
                rewards = rewards.squeeze(-1)
            if rewards.ndim == 1:
                # expand scalar reward per env to per-agent
                rewards = np.repeat(rewards[:, None], num_agents, axis=1)
            share_obs = share_obs if all_args.use_centralized_V else obs
            ep_rew += rewards
            if action_log is not None and ep == 0:
                # log only the first env's actions for brevity
                action_log.append({
                    "step": len(action_log),
                    "actions": actions_env[0, :num_agents].tolist(),
                    "rewards": rewards[0, :num_agents].tolist(),
                    "dones": dones[0].tolist() if hasattr(dones[0], "tolist") else bool(dones[0]),
                })

            if dones.ndim == 1:
                done_envs = dones
            else:
                done_envs = dones.any(axis=1)
            if done_envs.any():
                rnn_states[done_envs] = 0
                masks[done_envs] = 0
                # reset envs marked done
                obs_reset, share_obs_reset, available_actions_reset = envs.reset(done_envs)
                obs[done_envs] = obs_reset
                share_obs[done_envs] = share_obs_reset if all_args.use_centralized_V else obs_reset
                if available_actions is not None:
                    available_actions[done_envs] = available_actions_reset

        total_rewards.append(ep_rew.mean(axis=0))

    total_rewards = np.array(total_rewards)
    avg_rewards = total_rewards.mean(axis=0)
    print(f"Average rewards over {all_args.eval_episodes} episodes:")
    for aid, rew in enumerate(avg_rewards):
        print(f"  Agent {aid}: {rew:.3f}")
    if action_log is not None and len(action_log) > 0:
        import json
        with open(all_args.log_actions_path, "w") as f:
            for entry in action_log:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved step-by-step actions for first episode/env to {all_args.log_actions_path}")

    envs.close()


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # force IPPO-style decentralised critic for this eval
    all_args.use_centralized_V = False
    all_args.share_policy = False
    all_args.use_eval = True

    evaluate(all_args)


if __name__ == "__main__":
    main(sys.argv[1:])