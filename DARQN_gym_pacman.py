import numpy as np
from keras import layers, Model
import gymnasium as gym
import ale_py
import tensorflow as tf
import random
from collections import deque
import time
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
# HYPERPARAMETERS
# ------------------------------------------------------------------------

env_name = "ALE/Pacman-v5"
learning_rate = 0.0001
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
gamma = 0.99
batch_size = 32
memory_size = 50000
episodes = 500
update_target_frequency = 5
render_mode = None
loss_function = 'mse'
train_interval = 4
frame_stack = 4

# ------------------------------------------------------------------------
# SETUP ENVIRONMENT
# ------------------------------------------------------------------------

# Actions for Pacman:
# 0: NOOP (no action)
# 1: UP
# 2: RIGHT
# 3: LEFT
# 4: DOWN

env = gym.make(env_name, render_mode=render_mode)

observation_shape = env.observation_space.shape # (210, 160, 3)
action_space = int(env.action_space.n) # 5 actions

print(f"Observation shape: {observation_shape}")
print(f"Number of actions: {action_space}")

# Create frame stacking deque for each environment reset
frame_buffer = deque(maxlen=frame_stack)

# ------------------------------------------------------------------------

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

state_shape = (84, 84, frame_stack)

# ------------------------------------------------------------------------

os.makedirs('./saved_models', exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./saved_models/darqn_model_best.weights.h5',
        monitor='loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq='epoch'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=0
    )
]

# ATTENTION MECHANISM (CBAM)
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
# DARQN MODEL CREATION
# ------------------------------------------------------------------------

def create_darqn_model():
    """Create DARQN (Dueling Attention Recurrent Q-Network) model"""
    input_layer = layers.Input(shape=state_shape, name='input')
    
    # -------- CNN FEATURE EXTRACTION BRANCH --------
    
    conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu',  # Extract coarse features
                         padding='same', name='conv1')(input_layer)
    batch1 = layers.BatchNormalization()(conv1)                           # Normalize activations
    
    conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu',  # Extract mid-level features
                         padding='same', name='conv2')(batch1)
    batch2 = layers.BatchNormalization()(conv2)                           # Normalize activations
    
    conv3 = layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', # Extract fine-grained features
                         padding='same', name='conv3')(batch2)
    batch3 = layers.BatchNormalization()(conv3)                           # Normalize activations
    
    # -------- ATTENTION MECHANISM (CBAM) --------
    attention = CBAM(reduction_ratio=8, kernel_size=7)(batch3)           # Focus on important regions
    
    # -------- RECURRENT PROCESSING (TEMPORAL CONTEXT) --------
    flat = layers.Flatten()(attention)                                    # Flatten CNN output
    reshape_lstm = layers.Reshape((121, 128))(attention)                  # Reshape for LSTM input (11*11=121)
    lstm1 = layers.LSTM(256, return_sequences=True, activation='relu',    # Capture long-term dependencies
                       name='lstm1', dropout=0.2)(reshape_lstm)
    lstm2 = layers.LSTM(128, return_sequences=False, activation='relu',   # Final temporal encoding
                       name='lstm2', dropout=0.2)(lstm1)
    
    # -------- MERGE SPATIAL & TEMPORAL FEATURES --------
    merge = layers.Concatenate()([flat, lstm2])                           # Combine CNN + LSTM features
    
    # -------- DENSE PROCESSING LAYERS --------
    dense1 = layers.Dense(512, activation='relu', name='dense1')(merge)   # Rich feature representation
    dropout1 = layers.Dropout(0.3)(dense1)                               # Prevent overfitting
    
    dense2 = layers.Dense(256, activation='relu', name='dense2')(dropout1) # Refined representation
    dropout2 = layers.Dropout(0.2)(dense2)                               # Prevent overfitting
    
    # -------- DUELING ARCHITECTURE --------
    # Value Stream: estimates state value V(s)
    value_dense = layers.Dense(128, activation='relu',                   # Value processing layer
                              name='value_dense')(dropout2)
    value_output = layers.Dense(1, name='value')(value_dense)            # State value V(s)
    
    # Advantage Stream: estimates action advantages A(s,a)
    advantage_dense = layers.Dense(128, activation='relu',               # Advantage processing layer
                                  name='advantage_dense')(dropout2)
    advantage_output = layers.Dense(action_space, name='advantage')(     # Action advantages A(s,a)
        advantage_dense)
    
    # -------- DUELING COMBINATION --------
    # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    # Broadcast value to match advantage shape
    value_broadcast = layers.RepeatVector(action_space)(value_output)    # Repeat value for each action
    value_broadcast = layers.Reshape((action_space,))(value_broadcast)   # Reshape to match advantage
    
    # Normalize advantages
    advantage_mean = layers.Lambda(                                       # Calculate mean advantage
        lambda x: tf.reduce_mean(x, axis=1, keepdims=True)
    )(advantage_output)
    advantage_norm = layers.Subtract()([advantage_output, advantage_mean]) # Normalize
    
    # Combine value and normalized advantage
    q_output = layers.Add(name='q_values')([                             # Final Q-values output
        value_broadcast,
        advantage_norm
    ])
    
    model = Model(inputs=input_layer, outputs=q_output, name='DARQN')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_function
    )
    
    return model

q_model = create_darqn_model()
target_model = create_darqn_model()
target_model.set_weights(q_model.get_weights())
q_model.summary()

# ------------------------------------------------------------------------

memory = deque(maxlen=memory_size)

def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def sample_batch():
    """Sample a batch from memory"""
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int32),
        np.array(rewards, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(dones, dtype=np.float32)
    )

def epsilon_greedy_policy(state, epsilon):
    """Epsilon-greedy action selection"""
    if np.random.random() < epsilon:
        return np.random.choice(action_space)
    else:
        q_values = q_model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

# ------------------------------------------------------------------------
# TRAINING STEP
# ------------------------------------------------------------------------

def train_step():
    """Perform one training step"""
    if len(memory) < batch_size:
        return None
    
    states, actions, rewards, next_states, dones = sample_batch()
    
    # Compute target Q-values using target network
    next_q_values = target_model.predict(next_states, verbose=0)
    max_next_q_values = np.max(next_q_values, axis=1)
    
    # Compute target Q-values for training
    target_q_values = q_model.predict(states, verbose=0)
    for i, action in enumerate(actions):
        if dones[i]:
            target_q_values[i][action] = rewards[i]
        else:
            target_q_values[i][action] = rewards[i] + gamma * max_next_q_values[i]
    
    # Train on batch
    history = q_model.fit(
        states, target_q_values,
        verbose=0,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history.history['loss'][0] if history.history['loss'] else None


# ------------------------------------------------------------------------

reward_history = []
loss_history = []
training_steps_per_episode = []

# TRAINING LOOP
# ------------------------------------------------------------------------

for episode in range(episodes):
    start = time.time()
    obs, info = env.reset()
    state = initialize_frame_buffer(obs)
    
    total_reward = 0
    done = False
    steps = 0
    episode_losses = []
    training_steps = 0

    while not done:
        # Action selection
        action = epsilon_greedy_policy(state, epsilon)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = get_stacked_frames(obs)
        
        # Check if episode done
        done = terminated or truncated

        # Store transition (using environment reward directly)
        store_transition(state, action, reward, next_state, done)
        total_reward += reward

        state = next_state
        steps += 1
        
        # Train periodically
        if steps % train_interval == 0:
            loss = train_step()
            if loss is not None:
                episode_losses.append(loss)
                training_steps += 1

    end = time.time()
    elapsed_time = end - start
    
    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network
    if episode % update_target_frequency == 0:
        target_model.set_weights(q_model.get_weights())
    
    # Store metrics
    reward_history.append(total_reward)
    training_steps_per_episode.append(training_steps)
    
    # Calculate episode statistics
    avg_loss = np.mean(episode_losses) if episode_losses else 0
    min_loss = np.min(episode_losses) if episode_losses else 0
    max_loss = np.max(episode_losses) if episode_losses else 0
    loss_history.append(avg_loss)
    
    # Print episode info
    print(f"Episode: {episode+1}/{episodes} | Reward: {total_reward:.0f} | Eps: {epsilon:.3f} | Time: {elapsed_time:.2f}s | Steps: {steps} | Memory: {len(memory)}")
    print(f"  └─ Loss -> Avg: {avg_loss:.6f} | Min: {min_loss:.6f} | Max: {max_loss:.6f} | Trainings: {training_steps}")

    # Save model checkpoint every 20 episodes
    if (episode + 1) % 20 == 0:
        q_model.save_weights(f'./saved_models/darqn_model_episode_{episode+1}.weights.h5')
        print(f"\n{'='*100}")
        print(f"Model checkpoint saved at episode {episode+1}")
        print(f"Reward -> Best: {max(reward_history):.0f} | Avg (last 20): {np.mean(reward_history[-20:]):.0f}")
        print(f"Loss   -> Avg (last 20): {np.mean(loss_history[-20:]):.6f}")
        print(f"{'='*100}\n")

env.close()

# Save final model
q_model.save_weights('./saved_models/darqn_model_final.weights.h5')

# Export metrics to JSON for visualization
import json
os.makedirs('./metrics', exist_ok=True)
metrics_data = {
    'episodes': list(range(1, episodes + 1)),
    'reward_history': [float(r) for r in reward_history],
    'loss_history': [float(l) for l in loss_history],
    'training_steps_per_episode': [int(s) for s in training_steps_per_episode],
}
with open('./metrics/training_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=2)
print("\nMetrics exported to ./metrics/training_metrics.json")

# ------------------------------------------------------------------------
# TRAINING SUMMARY
# ------------------------------------------------------------------------

print("\n" + "="*100)
print("TRAINING COMPLETE - DARQN PACMAN")
print("="*100)
print(f"Total Episodes: {episodes}")
print(f"Best Reward: {max(reward_history):.0f}")
print(f"Average Reward: {np.mean(reward_history):.0f}")
print(f"Final Average Reward (last 20): {np.mean(reward_history[-20:]):.0f}")
print(f"Average Loss: {np.mean(loss_history):.6f}")
print(f"Final Average Loss (last 20): {np.mean(loss_history[-20:]):.6f}")
print(f"Total Training Steps: {sum(training_steps_per_episode)}")
print(f"Average Training Steps per Episode: {np.mean(training_steps_per_episode):.2f}")
print(f"Model saved to: ./saved_models/darqn_model_final.weights.h5")
print("="*100)