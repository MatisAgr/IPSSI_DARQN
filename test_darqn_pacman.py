import numpy as np
from keras import layers, Model
import gymnasium as gym
import ale_py
import tensorflow as tf
from collections import deque
import os

# ------------------------------------------------------------------------
# GPU CONFIGURATION
# ------------------------------------------------------------------------

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available(s): {len(gpus)}")
        print(f"GPU used: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, using CPU")

os.environ["OMP_NUM_THREADS"] = "14"
tf.config.threading.set_intra_op_parallelism_threads(14)
tf.config.threading.set_inter_op_parallelism_threads(14)

gym.register_envs(ale_py)

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------

env_name = "ALE/Pacman-v5"
model_path = "./saved_models/darqn_model_final.weights.h5"
render_mode = "human"
frame_stack = 4
num_test_episodes = 5

# ------------------------------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------------------------------

env = gym.make(env_name, render_mode=render_mode)
action_space = env.action_space.n
state_shape = (84, 84, frame_stack)

frame_buffer = deque(maxlen=frame_stack)

def preprocess_observation(obs):
    """Convert RGB to grayscale and resize to 84x84"""
    obs_gray = np.mean(obs, axis=2).astype(np.uint8)
    obs_resized = tf.image.resize(obs_gray[..., np.newaxis], (84, 84))
    return obs_resized.numpy().astype(np.float32) / 255.0

def get_stacked_frames(frame):
    """Stack frames for temporal context"""
    frame_buffer.append(preprocess_observation(frame))
    stacked = np.concatenate(list(frame_buffer), axis=-1)
    return stacked

def initialize_frame_buffer(initial_obs):
    """Initialize frame buffer with repeated frames"""
    frame_buffer.clear()
    processed = preprocess_observation(initial_obs)
    for _ in range(frame_stack):
        frame_buffer.append(processed)
    return np.concatenate(list(frame_buffer), axis=-1)

# ------------------------------------------------------------------------
# ATTENTION MECHANISMS
# ------------------------------------------------------------------------

class ChannelAttention(layers.Layer):
    """Channel Attention Module"""
    def __init__(self, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        
        self.fc1 = layers.Dense(max(1, channels // self.reduction_ratio), activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        avg_out = self.fc2(self.fc1(self.avg_pool(inputs)))
        max_out = self.fc2(self.fc1(self.max_pool(inputs)))
        return inputs * (avg_out + max_out)


class SpatialAttention(layers.Layer):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            1, self.kernel_size, padding='same', activation='sigmoid'
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        attention_map = self.conv(concat)
        return inputs * attention_map


class CBAM(layers.Layer):
    """Convolutional Block Attention Module"""
    def __init__(self, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

# ------------------------------------------------------------------------
# DARQN MODEL RECREATION
# ------------------------------------------------------------------------

def create_darqn_model():
    """Create DARQN model for inference"""
    input_layer = layers.Input(shape=state_shape, name='input')
    
    # CNN feature extraction branch
    conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', 
                         padding='same', name='conv1')(input_layer)
    batch1 = layers.BatchNormalization()(conv1)
    
    conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu',
                         padding='same', name='conv2')(batch1)
    batch2 = layers.BatchNormalization()(conv2)
    
    conv3 = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu',
                         padding='same', name='conv3')(batch2)
    batch3 = layers.BatchNormalization()(conv3)
    
    # Attention mechanism (CBAM)
    attention = CBAM(reduction_ratio=8, kernel_size=7)(batch3)
    
    # Flatten for recurrent processing
    flat = layers.Flatten()(attention)
    
    # Recurrent processing with LSTM for temporal context
    reshape_lstm = layers.Reshape((21, 128))(attention)
    lstm1 = layers.LSTM(256, return_sequences=True, activation='relu',
                       name='lstm1', dropout=0.2)(reshape_lstm)
    lstm2 = layers.LSTM(128, return_sequences=False, activation='relu',
                       name='lstm2', dropout=0.2)(lstm1)
    
    # Merge LSTM output with flattened CNN features
    merge = layers.Concatenate()([flat, lstm2])
    
    # Dense layers with dropout
    dense1 = layers.Dense(512, activation='relu', name='dense1')(merge)
    dropout1 = layers.Dropout(0.3)(dense1)
    
    dense2 = layers.Dense(256, activation='relu', name='dense2')(dropout1)
    dropout2 = layers.Dropout(0.2)(dense2)
    
    # Dueling Architecture
    # Value stream
    value_dense = layers.Dense(128, activation='relu', name='value_dense')(dropout2)
    value_output = layers.Dense(1, name='value')(value_dense)
    
    # Advantage stream
    advantage_dense = layers.Dense(128, activation='relu', name='advantage_dense')(dropout2)
    advantage_output = layers.Dense(action_space, name='advantage')(advantage_dense)
    
    # Combine value and advantage
    mean_advantage = layers.Lambda(
        lambda adv: adv - tf.reduce_mean(adv, axis=1, keepdims=True)
    )(advantage_output)
    
    q_output = layers.Add(name='q_values')([
        value_output,
        mean_advantage
    ])
    
    model = Model(inputs=input_layer, outputs=q_output, name='DARQN')
    return model

# Load model
print("Loading DARQN model...")
model = create_darqn_model()
model.load_weights(model_path)
print("Model loaded successfully!")

# ------------------------------------------------------------------------
# INFERENCE FUNCTION
# ------------------------------------------------------------------------

def greedy_action_selection(state):
    """Select action greedily (no exploration)"""
    q_values = model.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0]), np.max(q_values[0])

# ------------------------------------------------------------------------
# TESTING LOOP
# ------------------------------------------------------------------------

print(f"\nTesting DARQN on {num_test_episodes} episodes...")
print("="*100)

test_rewards = []

for episode in range(num_test_episodes):
    obs, info = env.reset()
    state = initialize_frame_buffer(obs)
    
    total_reward = 0
    done = False
    steps = 0
    
    while not done:
        action, q_value = greedy_action_selection(state)
        
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = get_stacked_frames(obs)
        
        done = terminated or truncated
        total_reward += reward
        
        state = next_state
        steps += 1
    
    test_rewards.append(total_reward)
    print(f"Test Episode {episode+1}/{num_test_episodes} | Reward: {total_reward:.0f} | Steps: {steps}")

print("="*100)
print(f"Average Test Reward: {np.mean(test_rewards):.0f}")
print(f"Max Test Reward: {np.max(test_rewards):.0f}")
print(f"Min Test Reward: {np.min(test_rewards):.0f}")
print("="*100)

env.close()
