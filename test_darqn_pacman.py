import numpy as np
from keras import layers, Model
import gymnasium as gym
import ale_py
import tensorflow as tf
from collections import deque
import os
import time
import argparse
import glob

# ------------------------------------------------------------------------
# COMMAND LINE ARGUMENTS
# ------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Test DARQN model on Pacman")
parser.add_argument("--checkpoint", type=int, default=None, help="Episode number to test")
parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration rate (0.0=greedy)")
parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes")
parser.add_argument("--render", action="store_true", help="Enable visual rendering")
parser.add_argument("--list", action="store_true", help="List all checkpoints")
args = parser.parse_args()

if args.list:
    print("\nAvailable checkpoints:")
    print("="*80)
    checkpoints = sorted(glob.glob("./saved_models/darqn_model_episode_*.weights.h5"))
    if checkpoints:
        for ckpt in checkpoints:
            ep_num = ckpt.split("_")[-1].replace(".weights.h5", "")
            size_mb = os.path.getsize(ckpt) / (1024**2)
            print(f"  Episode {ep_num:>4s} - {os.path.basename(ckpt)} ({size_mb:.1f} MB)")
    else:
        print("  No checkpoints found")
    print("="*80)
    exit(0)

if args.checkpoint is not None:
    weights_path = f"./saved_models/darqn_model_episode_{args.checkpoint}.weights.h5"
    if not os.path.exists(weights_path):
        print(f"Error: Checkpoint not found: {weights_path}")
        exit(1)
else:
    checkpoints = sorted(glob.glob("./saved_models/darqn_model_episode_*.weights.h5"))
    if checkpoints:
        weights_path = checkpoints[-1]
        ep_num = weights_path.split("_")[-1].replace(".weights.h5", "")
        print(f"Auto-detected latest checkpoint: Episode {ep_num}")
    else:
        print("Error: No checkpoints found")
        exit(1)

# ------------------------------------------------------------------------
# GPU CONFIGURATION
# ------------------------------------------------------------------------

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available(s): {len(gpus)}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, using CPU")

os.environ["OMP_NUM_THREADS"] = "14"
tf.config.threading.set_intra_op_parallelism_threads(14)
tf.config.threading.set_inter_op_parallelism_threads(14)

gym.register_envs(ale_py)

env_name = "ALE/Pacman-v5"
render_mode = "human" if args.render else None
frame_stack = 4
num_test_episodes = args.episodes
action_space = 5

env = gym.make(env_name, render_mode=render_mode)
state_shape = (84, 84, frame_stack)
frame_buffer = deque(maxlen=frame_stack)

def preprocess_observation(obs):
    obs_gray = np.mean(obs, axis=2).astype(np.uint8)
    obs_resized = tf.image.resize(obs_gray[..., np.newaxis], (84, 84))
    return obs_resized.numpy().astype(np.float32) / 255.0

def get_stacked_frames(frame):
    frame_buffer.append(preprocess_observation(frame))
    stacked = np.concatenate(list(frame_buffer), axis=-1)
    return stacked

def initialize_frame_buffer(initial_obs):
    frame_buffer.clear()
    processed = preprocess_observation(initial_obs)
    for _ in range(frame_stack):
        frame_buffer.append(processed)
    return np.concatenate(list(frame_buffer), axis=-1)

# ------------------------------------------------------------------------
# CBAM ATTENTION
# ------------------------------------------------------------------------

class CBAM(layers.Layer):
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(channels // self.ratio, 1)
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        self.fc1 = layers.Dense(hidden, activation="relu", kernel_initializer="he_normal", use_bias=True)
        self.fc2 = layers.Dense(channels, activation=None, kernel_initializer="he_normal", use_bias=True)
        self.spatial_conv = layers.Conv2D(filters=1, kernel_size=self.kernel_size, padding="same", activation="sigmoid", kernel_initializer="he_normal")
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = self.fc2(self.fc1(self.gap(inputs)))
        max_pool = self.fc2(self.fc1(self.gmp(inputs)))
        channel_attn = tf.nn.sigmoid(avg_pool + max_pool)
        channel_attn = tf.reshape(channel_attn, (-1, 1, 1, tf.shape(inputs)[-1]))
        x = inputs * channel_attn
        avg_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial = tf.concat([avg_spatial, max_spatial], axis=-1)
        spatial_attn = self.spatial_conv(spatial)
        return x * spatial_attn

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio, "kernel_size": self.kernel_size})
        return cfg

# ------------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------------

def create_darqn_model():
    input_layer = layers.Input(shape=state_shape, name="input")
    conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu", padding="same", name="conv1")(input_layer)
    batch1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu", padding="same", name="conv2")(batch1)
    batch2 = layers.BatchNormalization()(conv2)
    conv3 = layers.Conv2D(128, (3, 3), strides=(1, 1), activation="relu", padding="same", name="conv3")(batch2)
    batch3 = layers.BatchNormalization()(conv3)
    attention = CBAM(ratio=8, kernel_size=7)(batch3)
    flat = layers.Flatten()(attention)
    reshape_lstm = layers.Reshape((121, 128))(attention)
    lstm1 = layers.LSTM(256, return_sequences=True, activation="relu", name="lstm1", dropout=0.2)(reshape_lstm)
    lstm2 = layers.LSTM(128, return_sequences=False, activation="relu", name="lstm2", dropout=0.2)(lstm1)
    merge = layers.Concatenate()([flat, lstm2])
    dense1 = layers.Dense(512, activation="relu", name="dense1")(merge)
    dropout1 = layers.Dropout(0.3)(dense1)
    dense2 = layers.Dense(256, activation="relu", name="dense2")(dropout1)
    dropout2 = layers.Dropout(0.2)(dense2)
    value_dense = layers.Dense(128, activation="relu", name="value_dense")(dropout2)
    value_output = layers.Dense(1, name="value")(value_dense)
    advantage_dense = layers.Dense(128, activation="relu", name="advantage_dense")(dropout2)
    advantage_output = layers.Dense(action_space, name="advantage")(advantage_dense)
    value_broadcast = layers.RepeatVector(action_space)(value_output)
    value_broadcast = layers.Reshape((action_space,))(value_broadcast)
    advantage_mean = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage_output)
    advantage_norm = layers.Subtract()([advantage_output, advantage_mean])
    q_output = layers.Add(name="q_values")([value_broadcast, advantage_norm])
    model = Model(inputs=input_layer, outputs=q_output, name="DARQN")
    return model

print(f"\nLoading DARQN model from: {weights_path}")
print("="*100)
model = create_darqn_model()
model.load_weights(weights_path)
print(f"Model weights loaded successfully")
print(f"Model parameters: {model.count_params():,}")
print("="*100)

# ------------------------------------------------------------------------
# INFERENCE with training=False (CRITICAL FIX)
# ------------------------------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    if epsilon > 0 and np.random.random() < epsilon:
        return np.random.choice(action_space)
    else:
        q_values = model(state[np.newaxis], training=False).numpy()
        return np.argmax(q_values[0])

# ------------------------------------------------------------------------
# TESTING
# ------------------------------------------------------------------------

print(f"\nStarting {num_test_episodes} test episodes with epsilon={args.epsilon:.2f}")
if args.epsilon == 0.0:
    print("Mode: GREEDY (pure exploitation)")
else:
    print(f"Mode: EPSILON-GREEDY (exploration rate: {args.epsilon:.1%})")
print("="*100)

test_rewards = []
test_steps = []

for episode in range(num_test_episodes):
    start = time.time()
    obs, info = env.reset()
    state = initialize_frame_buffer(obs)
    total_reward = 0
    done = False
    steps = 0
    
    while not done:
        action = epsilon_greedy_action(state, args.epsilon)
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = get_stacked_frames(obs)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        steps += 1
    
    elapsed = time.time() - start
    test_rewards.append(total_reward)
    test_steps.append(steps)
    print(f"Episode {episode+1}/{num_test_episodes} | Reward: {total_reward:6.0f} | Steps: {steps:4d} | Time: {elapsed:6.2f}s")

print("="*100)
print(f"Average Test Reward:  {np.mean(test_rewards):6.1f}")
print(f"Max Test Reward:      {np.max(test_rewards):6.0f}")
print(f"Min Test Reward:      {np.min(test_rewards):6.0f}")
print(f"Std Reward:           {np.std(test_rewards):6.2f}")
print(f"Average Steps/Ep:     {np.mean(test_steps):6.1f}")
print("="*100)

env.close()
