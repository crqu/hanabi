    
import time
import wandb
import numpy as np
import torch

from onpolicy.runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class HanabiRunner(Runner):
    """Separated-policy runner for Hanabi (forward wrapper)."""

    def __init__(self, config):
        super(HanabiRunner, self).__init__(config)
        self.true_total_num_steps = 0

    def _slice_agent(self, arr, agent_id):
        """
        Handle returns that may already be per-agent (T, dim) or stacked (T, A, dim).
        """
        if arr is None:
            return None
        if arr.ndim >= 3:
            return arr[:, agent_id]
        return arr

    def run(self):
        # Pre-allocate per-agent turn buffers (shape matches shared runner style)
        obs_shape = self.buffer[0].obs.shape[2:]
        share_obs_shape = self.buffer[0].share_obs.shape[2:]
        act_shape = self.buffer[0].actions.shape[2:]
        avail_shape = self.buffer[0].available_actions.shape[2:] if self.buffer[0].available_actions is not None else None
        rnn_state_shape = self.buffer[0].rnn_states.shape[2:]
        rnn_state_critic_shape = self.buffer[0].rnn_states_critic.shape[2:]

        self.turn_obs = np.zeros((self.n_rollout_threads, self.num_agents, *obs_shape), dtype=np.float32)
        self.turn_share_obs = np.zeros((self.n_rollout_threads, self.num_agents, *share_obs_shape), dtype=np.float32)
        if avail_shape is not None:
            self.turn_available_actions = np.zeros((self.n_rollout_threads, self.num_agents, *avail_shape), dtype=np.float32)
        else:
            self.turn_available_actions = None
        self.turn_values = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer[0].value_preds.shape[2:]), dtype=np.float32)
        self.turn_actions = np.zeros((self.n_rollout_threads, self.num_agents, *act_shape), dtype=np.float32)
        self.turn_action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer[0].action_log_probs.shape[2:]), dtype=np.float32)
        self.turn_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, *rnn_state_shape), dtype=np.float32)
        self.turn_rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, *rnn_state_critic_shape), dtype=np.float32)
        self.turn_masks = np.ones((self.n_rollout_threads, self.num_agents, *self.buffer[0].masks.shape[2:]), dtype=np.float32)
        self.turn_active_masks = np.ones_like(self.turn_masks)
        self.turn_bad_masks = np.ones_like(self.turn_masks)
        self.turn_rewards = np.zeros((self.n_rollout_threads, self.num_agents, *self.buffer[0].rewards.shape[2:]), dtype=np.float32)
        self.turn_rewards_since_last_action = np.zeros_like(self.turn_rewards)

        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            self.scores = []
            for step in range(self.episode_length):
                self.reset_choose = np.zeros(self.n_rollout_threads) == 1.0
                # Sample actions and step env
                self.collect(step)

                if step == 0 and episode > 0:
                    # finalize previous episode's last step then train
                    for agent_id in range(self.num_agents):
                        self.buffer[agent_id].share_obs[-1] = self.turn_share_obs[:, agent_id].copy()
                        self.buffer[agent_id].obs[-1] = self.turn_obs[:, agent_id].copy()
                        if self.turn_available_actions is not None:
                            self.buffer[agent_id].available_actions[-1] = self.turn_available_actions[:, agent_id].copy()
                        self.buffer[agent_id].active_masks[-1] = self.turn_active_masks[:, agent_id].copy()

                        # shift rewards and append last step rewards
                        self.buffer[agent_id].rewards[0:self.episode_length-1] = self.buffer[agent_id].rewards[1:]
                        self.buffer[agent_id].rewards[-1] = self.turn_rewards[:, agent_id].copy()

                    self.compute()
                    train_infos = self.train()

                # insert turn data into buffers
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].chooseinsert(
                        self.turn_share_obs[:, agent_id],
                        self.turn_obs[:, agent_id],
                        self.turn_rnn_states[:, agent_id],
                        self.turn_rnn_states_critic[:, agent_id],
                        self.turn_actions[:, agent_id],
                        self.turn_action_log_probs[:, agent_id],
                        self.turn_values[:, agent_id],
                        self.turn_rewards[:, agent_id],
                        self.turn_masks[:, agent_id],
                        self.turn_bad_masks[:, agent_id],
                        self.turn_active_masks[:, agent_id],
                        None if self.turn_available_actions is None else self.turn_available_actions[:, agent_id],
                    )

                # env reset for done envs
                obs, share_obs, available_actions = self.envs.reset(self.reset_choose)
                share_obs = share_obs if self.use_centralized_V else obs
                self.use_obs[self.reset_choose] = obs[self.reset_choose]
                self.use_share_obs[self.reset_choose] = share_obs[self.reset_choose]
                if self.turn_available_actions is not None:
                    self.use_available_actions[self.reset_choose] = available_actions[self.reset_choose]

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if episode % self.log_interval == 0 and episode > 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.hanabi_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                average_score = np.mean(self.scores) if len(self.scores) > 0 else 0.0
                print("average score is {}.".format(average_score))
                if self.use_wandb:
                    wandb.log({'average_score': average_score}, step=self.true_total_num_steps)
                else:
                    self.writter.add_scalars('average_score', {'average_score': average_score}, self.true_total_num_steps)

                # aggregate rewards across agents
                mean_rewards = np.mean([np.mean(buf.rewards) for buf in self.buffer])
                if isinstance(train_infos, dict):
                    train_infos["average_step_rewards"] = mean_rewards
                    self.log_train(train_infos, self.true_total_num_steps)
                else:
                    # train_infos is a list (one dict per agent); add logging per agent
                    enriched_infos = []
                    for aid, info in enumerate(train_infos):
                        info = info if info is not None else {}
                        info["average_step_rewards"] = np.mean(self.buffer[aid].rewards)
                        enriched_infos.append(info)
                    self.log_train(enriched_infos, self.true_total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(self.true_total_num_steps)

    def warmup(self):
        # reset env
        self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
        obs, share_obs, available_actions = self.envs.reset(self.reset_choose)

        share_obs = share_obs if self.use_centralized_V else obs
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()

        for agent_id in range(self.num_agents):
            agent_share = self._slice_agent(share_obs, agent_id)
            agent_obs = self._slice_agent(obs, agent_id)
            self.buffer[agent_id].share_obs[0] = agent_share.copy()
            self.buffer[agent_id].obs[0] = agent_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        for current_agent_id in range(self.num_agents):
            env_actions = np.ones((self.n_rollout_threads, *self.turn_actions.shape[2:]), dtype=np.float32) * (-1.0)
            choose = np.any(self.use_available_actions == 1, axis=1)
            if ~np.any(choose):
                self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
                break

            # select actions for current agent
            self.trainer[current_agent_id].prep_rollout()
            agent_share_obs = self._slice_agent(self.use_share_obs[choose], current_agent_id)
            agent_obs = self._slice_agent(self.use_obs[choose], current_agent_id)
            agent_avail = self._slice_agent(self.use_available_actions[choose], current_agent_id) if self.use_available_actions is not None else None

            value, action, action_log_prob, rnn_state, rnn_state_critic = \
                self.trainer[current_agent_id].policy.get_actions(
                    agent_share_obs,
                    agent_obs,
                    self.turn_rnn_states[choose, current_agent_id],
                    self.turn_rnn_states_critic[choose, current_agent_id],
                    self.turn_masks[choose, current_agent_id],
                    agent_avail,
                )

            self.turn_obs[choose, current_agent_id] = agent_obs.copy()
            self.turn_share_obs[choose, current_agent_id] = agent_share_obs.copy()
            if self.turn_available_actions is not None:
                self.turn_available_actions[choose, current_agent_id] = agent_avail.copy()
            self.turn_values[choose, current_agent_id] = _t2n(value)
            self.turn_actions[choose, current_agent_id] = _t2n(action)
            env_actions[choose] = _t2n(action)
            self.turn_action_log_probs[choose, current_agent_id] = _t2n(action_log_prob)
            self.turn_rnn_states[choose, current_agent_id] = _t2n(rnn_state)
            self.turn_rnn_states_critic[choose, current_agent_id] = _t2n(rnn_state_critic)

            obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(env_actions)

            self.true_total_num_steps += (choose == True).sum()
            share_obs = share_obs if self.use_centralized_V else obs

            self.use_obs = obs.copy()
            self.use_share_obs = share_obs.copy()
            self.use_available_actions = available_actions.copy()

            # reward bookkeeping
            self.turn_rewards[choose, current_agent_id] = self.turn_rewards_since_last_action[choose, current_agent_id].copy()
            self.turn_rewards_since_last_action[choose, current_agent_id] = 0.0
            self.turn_rewards_since_last_action[choose] += rewards[choose]

            # handle done environments
            self.reset_choose[dones == True] = np.ones((dones == True).sum(), dtype=bool)
            if self.turn_available_actions is not None:
                self.use_available_actions[dones == True] = np.zeros(((dones == True).sum(), *self.turn_available_actions.shape[2:]), dtype=np.float32)
            self.turn_masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)
            self.turn_rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            self.turn_rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, *self.turn_rnn_states_critic.shape[2:]), dtype=np.float32)

            # active masks for current agent
            self.turn_active_masks[dones == True, current_agent_id] = np.ones(((dones == True).sum(), 1), dtype=np.float32)
            left_agent_id = current_agent_id + 1
            left_agents_num = self.num_agents - left_agent_id
            if left_agents_num > 0:
                self.turn_active_masks[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
                self.turn_rewards[dones == True, left_agent_id:] = self.turn_rewards_since_last_action[dones == True, left_agent_id:]
                self.turn_rewards_since_last_action[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
                self.turn_values[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
                self.turn_obs[dones == True, left_agent_id:] = 0
                self.turn_share_obs[dones == True, left_agent_id:] = 0

            # masks for non-done current agent
            self.turn_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)
            self.turn_active_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)

            for done, info in zip(dones, infos):
                if done and 'score' in info.keys():
                    self.scores.append(info['score'])

    @torch.no_grad()
    def eval(self, total_num_steps):
        # Placeholder: evaluation not used in current SLURM script
        pass
