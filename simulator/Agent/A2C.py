import os
import sys
import math
import time
import argparse
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # cleaner TF logs


def discounted_bootstrap_returns(rewards, dones, last_value, gamma):
    """
    Compute n-step bootstrapped returns G_t backward over a trajectory fragment.

    rewards: [T] float32
    dones:   [T] bool
    last_value: scalar V(s_{T}) for bootstrap
    returns: [T] float32
    """
    T = len(rewards)
    G = np.zeros_like(rewards, dtype=np.float32)
    running = last_value
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running * (1.0 - float(dones[t]))
        G[t] = running
    return G

def entropy_categorical(logits):
    # logits -> probabilities
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    return -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=-1))

# =========================================================
# Attention + MLP Actor-Critic Network
#   - First layer replaced by a self-attention block over feature tokens
#   - Then an MLP trunk; heads for policy and value
# =========================================================
class AttentionActorCritic(keras.Model):
    def __init__(self,
                 obs_dim,
                 n_actions,
                 d_model=64,
                 n_heads=4,
                 mlp_sizes=(128, 128),
                 attn_dropout=0.0,
                 ff_dropout=0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.d_model = d_model
        self.n_heads = n_heads

        # --- Tokenize features: (B, F) -> (B, F, d_model)
        # Map the flat observation to F tokens with d_model dims each
        self.to_tokens = keras.Sequential([
            layers.InputLayer(shape=(obs_dim,)),
            layers.Dense(obs_dim * d_model, activation=None),
            layers.Reshape((obs_dim, d_model)),
        ])

        # --- One self-attention block with residual + LayerNorm ---
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = layers.MultiHeadAttention(num_heads=n_heads,
                                              key_dim=d_model,
                                              dropout=attn_dropout)
        self.drop_attn = layers.Dropout(attn_dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn = keras.Sequential([
            layers.Dense(4 * d_model, activation=tf.nn.gelu),
            layers.Dropout(ff_dropout),
            layers.Dense(d_model, activation=None),
            layers.Dropout(ff_dropout),
        ])

        # --- Pool token features to a single vector ---
        self.pool = layers.GlobalAveragePooling1D()  # (B, F, d_model) -> (B, d_model)

        # --- MLP trunk after attention ---
        mlp_layers = []
        for h in mlp_sizes:
            mlp_layers += [layers.Dense(h, activation="tanh")]
        self.mlp = keras.Sequential(mlp_layers)

        # --- Policy & Value heads ---
        self.policy_logits = layers.Dense(n_actions, activation=None)
        self.value = layers.Dense(1, activation=None)

    def call(self, obs):
        # obs: (B, F)
        x = self.to_tokens(obs)                   # (B, F, d_model)
        # Self-attention block
        y = self.norm1(x)
        attn_out = self.attn(y, y, y)            # (B, F, d_model)
        x = x + self.drop_attn(attn_out)         # Residual
        y = self.norm2(x)
        x = x + self.ffn(y)                      # Residual

        # Pool & MLP
        x = self.pool(x)                          # (B, d_model)
        x = self.mlp(x)                           # (B, H)

        return self.policy_logits(x), self.value(x)

# =========================================================
# A2C Agent (synchronous)
# =========================================================
class A2C:
    def __init__(self, env_id="CartPole-v1",
                 gamma=0.99,
                 n_steps=5,               # rollout length
                 vf_coef=0.5,             # value loss coefficient
                 ent_coef=0.01,           # entropy bonus coefficient
                 lr=3e-4,
                 max_grad_norm=0.5,
                 seed=42,
                 # Attention net hyperparams
                 d_model=64,
                 n_heads=4,
                 mlp_sizes=(128, 128),
                 attn_dropout=0.0,
                 ff_dropout=0.0):

        self.env_id = env_id
        self.gamma = gamma
        self.n_steps = n_steps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        self.env = make_env(env_id, seed)
        self.eval_env = make_env(env_id, seed + 1)

        obs_space = self.env.observation_space
        act_space = self.env.action_space

        assert len(obs_space.shape) == 1, "This example supports 1D observations only."
        assert hasattr(act_space, "n"), "This example supports discrete actions."

        self.obs_dim = obs_space.shape[0]
        self.n_actions = act_space.n

        # Swap in our Attention + MLP policy/value network
        self.net = AttentionActorCritic(self.obs_dim,
                                        self.n_actions,
                                        d_model=d_model,
                                        n_heads=n_heads,
                                        mlp_sizes=mlp_sizes,
                                        attn_dropout=attn_dropout,
                                        ff_dropout=ff_dropout)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # For logging
        self.global_step = 0
        self.ep_returns = deque(maxlen=100)

    def select_action(self, obs_np):
        obs_tf = tf.convert_to_tensor(obs_np[None, :], dtype=tf.float32)
        logits, value = self.net(obs_tf)
        probs = tf.nn.softmax(logits)
        action = tf.random.categorical(tf.math.log(probs), 1)
        return int(action[0, 0].numpy()), float(value[0, 0].numpy())

    def evaluate_value(self, obs_np):
        obs_tf = tf.convert_to_tensor(obs_np[None, :], dtype=tf.float32)
        _, value = self.net(obs_tf)
        return float(value[0, 0].numpy())

    @tf.function
    def train_step(self, obs, actions, returns, advantages):
        """
        One gradient step on a minibatch.
        obs: [B, obs_dim]
        actions: [B] int32
        returns: [B] float32 (bootstrapped targets)
        advantages: [B] float32 (G_t - V(s_t))
        """
        with tf.GradientTape() as tape:
            logits, values = self.net(obs)
            values = tf.squeeze(values, axis=-1)  # [B]

            # Policy loss
            log_probs = tf.nn.log_softmax(logits)
            act_one_hot = tf.one_hot(actions, depth=logits.shape[-1], dtype=tf.float32)
            log_pi_a = tf.reduce_sum(act_one_hot * log_probs, axis=-1)  # [B]
            policy_loss = -tf.reduce_mean(log_pi_a * tf.stop_gradient(advantages))

            # Value loss (MSE)
            value_loss = tf.reduce_mean(tf.square(returns - values))

            # Entropy bonus (maximize entropy => minimize -entropy)
            ent = entropy_categorical(logits)
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent

        grads = tape.gradient(loss, self.net.trainable_variables)
        # Clip global norm
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
        return policy_loss, value_loss, ent, loss

    def rollout_n_steps(self, env, start_obs):
        """
        Collect a trajectory fragment of length <= n_steps (stops early on terminal).
        Returns: dict with obs, actions, rewards, dones, last_obs, last_done, last_value
        """
        obs_list, act_list, rew_list, done_list, val_list = [], [], [], [], []
        obs = start_obs
        ep_return = 0.0
        for t in range(self.n_steps):
            action, value = self.select_action(obs)
            if new_gym_api:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:
                next_obs, reward, done, info = env.step(action)
            obs_list.append(obs.copy())
            act_list.append(action)
            rew_list.append(reward)
            done_list.append(done)
            val_list.append(value)
            ep_return += reward

            obs = next_obs
            self.global_step += 1

            if done:
                if new_gym_api:
                    next_obs, _ = env.reset()
                else:
                    next_obs = env.reset()
                self.ep_returns.append(ep_return)
                ep_return = 0.0
            # continue even if done to fill n_steps (A2C often does per-env reset)

        # Bootstrap from last state
        last_value = 0.0 if done_list and done_list[-1] else self.evaluate_value(obs)

        return {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(act_list, dtype=np.int32),
            "rewards": np.array(rew_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=np.bool_),
            "last_obs": obs,
            "last_value": last_value,
        }

    def train(self, total_steps=200_000, batch_updates_per_rollout=1,
              log_interval=1000, checkpoint_dir="checkpoints/a2c_cartpole_attn"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initial reset
        if new_gym_api:
            obs, _ = self.env.reset()
        else:
            obs = self.env.reset()

        last_log_step = 0
        best_avg = -1e9
        ckpt = tf.train.Checkpoint(model=self.net, optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

        while self.global_step < total_steps:
            # Collect n-step rollout
            traj = self.rollout_n_steps(self.env, obs)
            obs = traj["last_obs"]

            # Compute returns and advantages
            returns = discounted_bootstrap_returns(traj["rewards"], traj["dones"], traj["last_value"], self.gamma)
            # Recompute V(s_t) for consistency
            obs_tf = tf.convert_to_tensor(traj["obs"], dtype=tf.float32)
            with tf.device("/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"):
                logits, values_tf = self.net(obs_tf)
            values = values_tf.numpy().squeeze(-1).astype(np.float32)
            advantages = returns - values

            # Normalize advantages
            adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # Train step
            policy_loss, value_loss, ent, total_loss = self.train_step(
                tf.convert_to_tensor(traj["obs"], dtype=tf.float32),
                tf.convert_to_tensor(traj["actions"], dtype=tf.int32),
                tf.convert_to_tensor(returns, dtype=tf.float32),
                tf.convert_to_tensor(advantages, dtype=tf.float32),
            )

            # Logging
            if self.global_step - last_log_step >= log_interval:
                last_log_step = self.global_step
                avg_return = np.mean(self.ep_returns) if self.ep_returns else float("nan")
                print(f"[step {self.global_step:7d}] "
                      f"avg_return={avg_return:7.2f}  "
                      f"loss={float(total_loss):.4f}  "
                      f"pi={float(policy_loss):.4f}  "
                      f"vf={float(value_loss):.4f}  "
                      f"ent={float(ent):.4f}")
                # Save best
                if not np.isnan(avg_return) and avg_return > best_avg:
                    best_avg = avg_return
                    manager.save()

        # Final save
        manager.save()
        print("Training complete. Checkpoints at:", checkpoint_dir)

    def evaluate(self, episodes=10, checkpoint=None, render=False):
        if checkpoint:
            ckpt = tf.train.Checkpoint(model=self.net)
            ckpt.restore(tf.train.latest_checkpoint(checkpoint)).expect_partial()
            print("Loaded checkpoint:", tf.train.latest_checkpoint(checkpoint))

        returns = []
        for ep in range(episodes):
            if new_gym_api:
                obs, _ = self.eval_env.reset()
            else:
                obs = self.eval_env.reset()
            done = False
            total_r = 0.0
            while not done:
                obs_tf = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
                logits, v = self.net(obs_tf)
                probs = tf.nn.softmax(logits)
                action = tf.argmax(probs, axis=-1)[0].numpy()  # greedy
                if new_gym_api:
                    obs, reward, terminated, truncated, info = self.eval_env.step(int(action))
                    done = terminated or truncated
                else:
                    obs, reward, done, info = self.eval_env.step(int(action))
                total_r += reward
                if render:
                    self.eval_env.render()
            returns.append(total_r)
            print(f"Episode {ep+1}: return = {total_r:.2f}")
        print(f"Avg return over {episodes} episodes: {np.mean(returns):.2f}")

# =========================================================
# Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="A2C (TF2) with Attention+MLP on CartPole-v1")
    p.add_argument("--env", type=str, default="CartPole-v1", help="Gym id")
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--episodes", type=int, default=10, help="eval episodes")
    p.add_argument("--checkpoint", type=str, default="checkpoints/a2c_cartpole_attn")
    p.add_argument("--render", action="store_true")
    # Attention-specific args
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--mlp-hidden", type=int, nargs='+', default=[128, 128])
    p.add_argument("--attn-dropout", type=float, default=0.0)
    p.add_argument("--ff-dropout", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    agent = A2C(env_id=args.env,
                gamma=args.gamma,
                n_steps=args.n_steps,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                lr=args.lr,
                max_grad_norm=args.max_grad_norm,
                seed=42,
                d_model=args.d_model,
                n_heads=args.n_heads,
                mlp_sizes=tuple(args.mlp_hidden),
                attn_dropout=args.attn_dropout,
                ff_dropout=args.ff_dropout)
    if args.train:
        agent.train(total_steps=args.total_steps,
                    log_interval=args.log_interval,
                    checkpoint_dir=args.checkpoint)
    if args.eval:
        agent.evaluate(episodes=args.episodes,
                       checkpoint=args.checkpoint,
                       render=args.render)


if __name__ == "__main__":
    main()
