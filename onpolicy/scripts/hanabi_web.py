#!/usr/bin/env python
"""
Simple Hanabi web server: play as agent 0 against a trained policy (agent 1).
Backend: FastAPI; Frontend: static HTML/JS served from /web/hanabi/index.html.

Usage:
  python -m onpolicy.scripts.serve_hanabi_web \
    --actor_path_b /path/to/actor_agent1.pt \
    --hanabi_name Hanabi-Full --num_agents 2 --hidden_size 512 --layer_N 2 \
    --host 0.0.0.0 --port 8000
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from onpolicy.config import get_config
from onpolicy.envs.hanabi.Hanabi_Env import HanabiEnv
from onpolicy.envs.hanabi import pyhanabi
from onpolicy.envs.env_wrappers import ChooseDummyVecEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


def make_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = HanabiEnv(all_args, all_args.seed + rank * 1000)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return ChooseDummyVecEnv([get_env_fn(0)])


def load_actor(all_args, envs, actor_path, device, use_recurrent):
    tmp_args = argparse.Namespace(**vars(all_args))
    tmp_args.use_recurrent_policy = use_recurrent
    tmp_args.use_naive_recurrent_policy = use_recurrent
    tmp_args.use_centralized_V = False
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
    return policy.actor, use_recurrent


class HanabiSession:
    def __init__(self, all_args, actor_path_ai, use_recurrent_ai, device):
        self.all_args = all_args
        self.device = device
        self.envs = make_env(all_args)
        self.num_agents = all_args.num_agents
        self.ai_id = 1  # human is 0, AI is 1
        self.actor_ai, self.ai_recurrent = load_actor(all_args, self.envs, actor_path_ai, device, use_recurrent_ai)
        self.last_ai_action = None
        # init rnn states
        self.reset_states()
        self.reset_env()

    def reset_states(self):
        self.rnn_states = np.zeros((1, self.num_agents, self.all_args.recurrent_N, self.all_args.hidden_size), dtype=np.float32)
        self.masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

    def reset_env(self):
        reset_choose = np.ones(1, dtype=bool)
        obs, share_obs, available_actions = self.envs.reset(reset_choose)
        self.obs = obs
        self.share_obs = share_obs if self.all_args.use_centralized_V else obs
        self.available_actions = available_actions
        self.last_ai_action = None
        self.reset_states()
        # If AI starts, auto-play to human turn.
        self._auto_play_ai_until_human()
        return self._build_state()

    def _select_ai_action(self):
        agent_obs = self.obs[:, self.ai_id] if self.obs.ndim >= 3 else self.obs
        agent_avail = None
        if self.available_actions is not None:
            if self.available_actions.ndim >= 3:
                agent_avail = self.available_actions[:, self.ai_id]
            else:
                agent_avail = self.available_actions
        actor = self.actor_ai
        action, _, rnn_state = actor(
            agent_obs,
            self.rnn_states[:, self.ai_id],
            self.masks[:, self.ai_id],
            agent_avail,
            deterministic=True,
        )
        self.rnn_states[:, self.ai_id] = rnn_state.cpu().numpy()
        act_np = action.detach().cpu().numpy()
        if act_np.ndim > 1:
            act_np = act_np.squeeze(-1)
        return int(act_np[0])

    def _step_env(self, action_int):
        """Step env with a single action int for the current player."""
        obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(np.array([[action_int]]))
        self.obs = obs
        self.share_obs = share_obs if self.all_args.use_centralized_V else obs
        self.available_actions = available_actions
        return rewards, dones, infos

    def _auto_play_ai_until_human(self):
        """If it's AI's turn, keep playing until human's turn or terminal."""
        env_backend = self.envs.envs[0]
        total_rewards = np.zeros((1, self.num_agents))
        infos = None
        dones = np.array([False])
        while env_backend.state.cur_player() == self.ai_id:
            ai_action = self._select_ai_action()
            self.last_ai_action = int(ai_action)
            r, d, infos = self._step_env(ai_action)
            total_rewards += np.asarray(r).reshape(total_rewards.shape)
            if d.any():
                dones = d
                break
        return total_rewards, dones, infos

    def step(self, human_action):
        """Human acts on their turn; AI auto-plays its turns."""
        env_backend = self.envs.envs[0]
        cum_rewards = np.zeros((1, self.num_agents))
        infos = None
        dones = np.array([False])

        # If it's AI turn, auto-play to human turn.
        if env_backend.state.cur_player() == self.ai_id:
            r, d, infos = self._auto_play_ai_until_human()
            cum_rewards += r
            if d.any():
                dones = d

        if not dones.any():
            # Human action (only when it's human turn)
            r, d, infos = self._step_env(human_action)
            cum_rewards += np.asarray(r).reshape(cum_rewards.shape)
            dones = d
            # After human move, let AI play until it is human turn again or game ends.
            if not dones.any():
                r2, d2, infos = self._auto_play_ai_until_human()
                cum_rewards += r2
                if d2.any():
                    dones = d2

        # handle episode end
        if dones.ndim == 1:
            done_envs = dones
        else:
            done_envs = dones.any(axis=1)
        if done_envs.any():
            self.reset_states()
            reset_choose = done_envs.astype(bool)
            obs_reset, share_obs_reset, available_actions_reset = self.envs.reset(reset_choose)
            self.obs[done_envs] = obs_reset
            self.share_obs[done_envs] = share_obs_reset if self.all_args.use_centralized_V else obs_reset
            if self.available_actions is not None:
                self.available_actions[done_envs] = available_actions_reset
        return {
            "state": self._build_state(),
            "rewards": cum_rewards[0].tolist(),
            "done": bool(done_envs[0]),
            "actions": [human_action, self.last_ai_action],
            "actions_text": [
                self._action_text(human_action, actor_id=0),
                self._action_text(self.last_ai_action, actor_id=self.ai_id) if self.last_ai_action is not None else None,
            ],
        }

    def _action_text(self, uid, actor_id):
        """Convert action id to friendly text using the same mapping as legal moves."""
        if uid is None:
            return None
        env_backend = self.envs.envs[0]
        try:
            move = env_backend.game.get_move(int(uid))
        except Exception:
            return str(uid)
        mtype = move.type()
        if mtype == pyhanabi.HanabiMoveType.PLAY:
            return f"Play card {move.card_index() + 1}"
        if mtype == pyhanabi.HanabiMoveType.DISCARD:
            return f"Discard card {move.card_index() + 1}"
        if mtype == pyhanabi.HanabiMoveType.REVEAL_COLOR:
            color_char = pyhanabi.color_idx_to_char(move.color())
            target = (actor_id + move.target_offset()) % self.num_agents
            pos = self._hint_positions(actor_id, move)
            pos_text = f" (cards {pos})" if pos else ""
            return f"Hint color {self._color_full(color_char)} to {self._player_name(target)}{pos_text}"
        if mtype == pyhanabi.HanabiMoveType.REVEAL_RANK:
            target = (actor_id + move.target_offset()) % self.num_agents
            pos = self._hint_positions(actor_id, move)
            pos_text = f" (cards {pos})" if pos else ""
            return f"Hint rank {move.rank() + 1} to {self._player_name(target)}{pos_text}"
        return str(move)

    def _color_full(self, c):
        return {"R": "Red", "Y": "Yellow", "G": "Green", "W": "White", "B": "Blue"}.get(c, "?")

    def _player_name(self, pid):
        if pid == self.ai_id:
            return "AI"
        if pid == 0:
            return "You"
        return f"Player {pid}"

    def _hint_positions(self, actor_id, move):
        """Return 1-based card positions (from actor's perspective) that match the hinted color/rank."""
        try:
            obs = self.envs.envs[0].state.observation(actor_id)
            target_offset = move.target_offset()
            target_hand = list(obs.observed_hands()[target_offset])
        except Exception:
            return []
        positions = []
        for idx, card in enumerate(target_hand):
            if move.type() == pyhanabi.HanabiMoveType.REVEAL_COLOR and card.color() == move.color():
                positions.append(idx + 1)
            if move.type() == pyhanabi.HanabiMoveType.REVEAL_RANK and card.rank() == move.rank():
                positions.append(idx + 1)
        return positions

    def _legal_moves_human(self, env_backend, legal_uids, actor_id=0):
        """Convert move ids to readable strings and details."""
        moves = []
        for uid in legal_uids:
            move = env_backend.game.get_move(uid)
            mtype = move.type()
            text = str(move)
            # Friendlier text
            if mtype == pyhanabi.HanabiMoveType.PLAY:
                text = f"Play card {move.card_index() + 1}"
            elif mtype == pyhanabi.HanabiMoveType.DISCARD:
                text = f"Discard card {move.card_index() + 1}"
            elif mtype == pyhanabi.HanabiMoveType.REVEAL_COLOR:
                color_char = pyhanabi.color_idx_to_char(move.color())
                target = (actor_id + move.target_offset()) % self.num_agents
                pos = self._hint_positions(actor_id, move)
                pos_text = f" (cards {pos})" if pos else ""
                text = f"Hint color {self._color_full(color_char)} to {self._player_name(target)}{pos_text}"
            elif mtype == pyhanabi.HanabiMoveType.REVEAL_RANK:
                target = (actor_id + move.target_offset()) % self.num_agents
                pos = self._hint_positions(actor_id, move)
                pos_text = f" (cards {pos})" if pos else ""
                text = f"Hint rank {move.rank() + 1} to {self._player_name(target)}{pos_text}"
            moves.append({
                "id": int(uid),
                "text": text,
                "raw": str(move),
                "type": int(mtype),
                "card_index": move.card_index(),
                "target_offset": move.target_offset(),
                "color": move.color(),
                "rank": move.rank(),
            })
        return moves

    def _build_state(self):
        # Use backend observation for readable info
        env_backend = self.envs.envs[0]
        obs_full = env_backend._make_observation_all_players()  # pylint: disable=protected-access
        human_obs = obs_full["player_observations"][0]
        legal_uids = human_obs.get("legal_moves_as_int", [])

        def card_to_str(card):
            color = card.get("color")
            rank = card.get("rank")
            color_str = self._color_full(color) if color else "Unknown"
            if rank is None or rank < 0:
                rank_str = "Unknown"
            else:
                rank_str = str(rank + 1)
            return f"{color_str} {rank_str}"

        observed_hands = []
        for hand in human_obs.get("observed_hands", []):
            observed_hands.append([card_to_str(c) for c in hand])

        discard_pile = [card_to_str(c) for c in human_obs.get("discard_pile", [])]

        fireworks = {}
        if human_obs.get("fireworks"):
            fireworks = {self._color_full(k): v for k, v in human_obs["fireworks"].items()}

        return {
            "current_player": obs_full["current_player"],
            "life_tokens": human_obs.get("life_tokens"),
            "information_tokens": human_obs.get("information_tokens"),
            "fireworks": fireworks,
            "discard_pile": discard_pile,
            "observed_hands": observed_hands,
            "legal_moves": self._legal_moves_human(env_backend, legal_uids, actor_id=0),
        }


def build_app(all_args):
    actor_path_ai = all_args.actor_path_b  # agent1 policy
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Locate frontend directory
    if all_args.web_dir is not None:
        static_dir = Path(all_args.web_dir)
    else:
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "web" / "hanabi",
            Path.cwd() / "web" / "hanabi",
        ]
        static_dir = next((p for p in candidates if p.exists()), None)
    if static_dir and static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")
    session = HanabiSession(all_args, actor_path_ai, all_args.use_recurrent_b, device)

    @app.get("/")
    async def index():
        if static_dir:
            index_path = static_dir / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
        return {"message": "Frontend not found. Please open /static/index.html manually via /static/index.html or set --web_dir."}

    @app.post("/reset")
    async def reset():
        state = session.reset_env()
        return {"state": state}

    @app.post("/step")
    async def step(payload: dict):
        if "action" not in payload:
            raise HTTPException(status_code=400, detail="Missing action")
        try:
            act = int(payload["action"])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid action")
        result = session.step(act)
        return result

    return app


def main():
    parser = get_config()
    parser.add_argument("--actor_path_b", type=str, required=True, help="Path to AI actor checkpoint (agent1)")
    parser.add_argument("--use_recurrent_b", action="store_true", default=False, help="Whether AI actor uses RNN")
    parser.add_argument("--hanabi_name", type=str, default="Hanabi-Full", help="Which Hanabi env to run")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of players (you are agent0)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web_dir", type=str, default=None, help="Path to web frontend (defaults to on-policy/web/hanabi)")
    args = parser.parse_args()
    # force decentralized, separate policies
    args.env_name = "Hanabi"
    args.use_centralized_V = False
    args.share_policy = False
    app = build_app(args)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
