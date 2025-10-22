# TODO: Dueling DQN for Pac-Man

import numpy as np
from keras import layers, Model
import gymnasium as gym # maj vers gymnasium
import ale_py # Support pour ALE/Pac-Man
import tensorflow as tf
import random
from collections import deque
import time

import os


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible(s): {len(gpus)}")
        print(f"GPU utilisé: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("Aucun GPU détecté, utilisation du CPU")


os.environ["OMP_NUM_THREADS"] = "14" 
tf.config.threading.set_intra_op_parallelism_threads(14)
tf.config.threading.set_inter_op_parallelism_threads(14)

# Enregistrer les jeux ALE
gym.register_envs(ale_py)

# ------------------------------------------------------------------------

# Pac-Man Actions:
# 0: NOOP (no operation)
# 1: UP
# 2: RIGHT
# 3: DOWN
# 4: LEFT
# 5: EAT PELLET (special)

# ------------------------------------------------------------------------


env_name = "ALE/Pacman-v5"      # env name ALE (Pac-Man)
learning_rate = 0.0001          # learning rate (reduced for stability)
epsilon = 1.0                   # exploration rate
epsilon_min = 0.2               # min exploration rate
epsilon_decay = 0.995           # decay rate
gamma = 0.99                    # discount factor (higher for Pac-Man)
batch_size = 64                 # batch size
memory_size = 50000             # replay memory size (larger for Pac-Man complexity)
episodes = 200                  # number of episodes (more for complex game)
update_target_frequency = 5     # update target model every 5 episodes (more frequent)
render_mode = None              # "human" for UI
loss = 'huber'                  # loss function (Huber for stability)
train_interval = 4              # train every 4 steps


# Pac-Man specific custom reward variables
pellet_eaten_reward = 10.0       # reward for eating a pellet
power_pellet_reward = 50.0       # reward for eating a power pellet
ghost_eaten_reward = 200.0       # reward for eating a ghost
caught_penalty = -100.0          # penalty for being caught
win_reward = 500.0               # reward for winning

survival_reward = 0.1            # reward per step (small survival bonus)
game_over_penalty = -50.0        # penalty for game over

positive_reward_multiplier = 2.0 # multiplier for positive original rewards
negative_reward_multiplier = 1.0 # multiplier for negative original rewards
score_increase_divisor = 10      # divisor to estimate pellets eaten from score

# ------------------------------------------------------------------------

env = gym.make(env_name, render_mode=render_mode)

observation_shape = env.observation_space.shape  # (210, 160, 3)
action_space = env.action_space.n  # 5 actions

def preprocess_observation(obs):
    """convert RGB to greyscale and resize to 84x84"""
    obs_gray = np.mean(obs, axis=2)
    obs_resized = tf.image.resize(obs_gray[..., np.newaxis], (84, 84))
    return obs_resized.numpy().flatten().astype(np.float32)

state_shape = preprocess_observation(env.reset()[0]).shape[0]
action_shape = action_space
print(f"state flat: {state_shape}")
print(f"Actions: {action_shape}")

# ------------------------------------------------------------------------

os.makedirs('./saved_models', exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True,
        verbose=0
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./saved_models/dqn_model_best.weights.h5',
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
        patience=3,
        min_lr=1e-6,
        verbose=0
    )
]

# ------------------------------------------------------------------------

# Dueling DQN Architecture
def create_dueling_q_model():
    """
    Dueling DQN: Separates value function V(s) and advantage function A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    
    Benefits:
    - Better convergence on complex games like Pac-Man
    - More stable learning
    - Learns value of being in a state independently of actions
    """
    input_layer = layers.Input(shape=(state_shape,))
    
    # Shared layers
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Value stream (V(s))
    value_stream = layers.Dense(64, activation='relu')(x)
    value_stream = layers.Dropout(0.2)(value_stream)
    value = layers.Dense(1, activation=None)(value_stream)
    
    # Advantage stream (A(s,a))
    advantage_stream = layers.Dense(64, activation='relu')(x)
    advantage_stream = layers.Dropout(0.2)(advantage_stream)
    advantages = layers.Dense(action_shape, activation=None)(advantage_stream)
    
    # Combine: Q(s,a) = V(s) + (A(s,a) - mean_A)
    # Broadcasting: V has shape (batch, 1), advantages has shape (batch, action_shape)
    mean_advantages = layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantages)
    q_values = layers.Add()([value, layers.Subtract()([advantages, mean_advantages])])
    
    model = Model(inputs=input_layer, outputs=q_values)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss,
    )
    return model

q_model = create_dueling_q_model()
target_model = create_dueling_q_model()
target_model.set_weights(q_model.get_weights())

print("\n" + "="*100)
print("Dueling DQN Architecture Created for Pac-Man")
print("="*100)
q_model.summary()
print("="*100 + "\n")


# ------------------------------------------------------------------------

memory = deque(maxlen=memory_size)

def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# sample a batch from memory
def sample_batch():
    batch = random.sample(memory, batch_size) # 32
    state, action, reward, next_state, done = map(np.asarray, zip(*batch))
    return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
def epsilon_greedy_policy(state, epsilon):
    if np.random.random() < epsilon:
        # random action (exploration)
        return np.random.choice(action_shape)
    else:
        # best action according to the Q model (exploitation)
        q_values = q_model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])


# ------------------------------------------------------------------------

def train_step():
    if len(memory) < batch_size:
        return
    state, action, reward, next_state, done = sample_batch()

    # Forward propagation
    next_q_values = target_model(next_state, training=False)
    max_next_q_values = np.max(next_q_values, axis=1)

    target_q_values = q_model(state, training=False).numpy()
    for i, act in enumerate(action):
        target_q_values[i][act] = reward[i] if done[i] else reward[i] + gamma * max_next_q_values[i]

    # Train with verbose to capture loss
    history = q_model.fit(state, target_q_values, verbose=0, callbacks=callbacks, batch_size=batch_size)
    
    return history.history['loss'][0] if history.history['loss'] else None


# ------------------------------------------------------------------------

def calculate_custom_reward(info, prev_info, reward, done):
    """Custom reward for Pac-Man"""
    custom_reward = 0
    
    # penalty if caught or lost
    if done and reward < 0:
        custom_reward += caught_penalty
        return custom_reward
    
    # win condition
    if done and reward > 0:
        custom_reward += win_reward
        return custom_reward
    
    # extract information if available
    current_score = info.get('score', 0)
    prev_score = prev_info.get('score', 0) if prev_info else 0
    
    # Reward based on score increase (eating pellets)
    score_increase = current_score - prev_score
    
    if score_increase > 0:
        # Estimate pellets eaten from score increase
        pellets_eaten = score_increase // score_increase_divisor
        
        if score_increase >= 200:  # Power pellet (typically worth 200)
            custom_reward += power_pellet_reward
        elif score_increase >= 50:  # Ghost eaten (typically worth 50-200)
            custom_reward += ghost_eaten_reward
        else:  # Regular pellets
            custom_reward += pellet_eaten_reward * pellets_eaten
    
    # survival reward (encourages longer games)
    custom_reward += survival_reward
    
    # Bonus for positive original reward
    if reward > 0:
        custom_reward += reward * positive_reward_multiplier
    
    # Penalty for negative original reward
    if reward < 0:
        custom_reward += reward * negative_reward_multiplier
    
    return custom_reward

# ------------------------------------------------------------------------

reward_history = []
loss_history = []
training_steps_per_episode = []

for episode in range(episodes):
    start = time.time()
    obs, info = env.reset()
    state = preprocess_observation(obs)
    total_reward = 0
    total_custom_reward = 0
    done = False
    steps = 0
    prev_info = info.copy()
    episode_losses = []
    training_steps = 0

    while not done:
        action = epsilon_greedy_policy(state, epsilon)
        
        # execute action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess_observation(obs)
        
        # determine if episode is done
        done = terminated or truncated

        # calculate custom reward
        custom_reward = calculate_custom_reward(info, prev_info, reward, done)
        
        # keep only custom reward in memory
        store_transition(state, action, custom_reward, next_state, done)
        total_reward += reward
        total_custom_reward += custom_reward

        state = next_state
        prev_info = info.copy()
        steps += 1
        
        # train less 
        if steps % train_interval == 0:
            loss = train_step()
            if loss is not None:
                episode_losses.append(loss)
                training_steps += 1

    end = time.time()
    timelength = end - start
        
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % update_target_frequency == 0:
        target_model.set_weights(q_model.get_weights())
    
    reward_history.append(total_custom_reward)
    training_steps_per_episode.append(training_steps)
    
    # calculate episode stats
    avg_loss = np.mean(episode_losses) if episode_losses else 0
    min_loss = np.min(episode_losses) if episode_losses else 0
    max_loss = np.max(episode_losses) if episode_losses else 0
    loss_history.append(avg_loss)
    
    # stats
    print(f"Episode: {episode}/{episodes} | Original: {total_reward:.0f} | Custom: {total_custom_reward:.0f} | Epsilon: {epsilon:.3f} | Time: {timelength:.2f}s | Steps: {steps} | Memory: {len(memory)}")
    print(f"  └─ Loss → Avg: {avg_loss:.6f} | Min: {min_loss:.6f} | Max: {max_loss:.6f} | Trainings: {training_steps}")

    # save each 20 epoch
    if (episode + 1) % 20 == 0:
        q_model.save_weights(f'./saved_models/dqn_model_episode_{episode+1}.weights.h5')
        print(f"\n{'='*100}")
        print(f"Model save epoch n {episode+1}")
        print(f"Reward → Best: {max(reward_history):.0f} | Avg (20 last): {np.mean(reward_history[-20:]):.0f}")
        print(f"Loss   → Avg (20 last): {np.mean(loss_history[-20:]):.6f}")
        print(f"{'='*100}\n")

env.close()

q_model.save_weights('./saved_models/dqn_model_final.weights.h5')

# ------------------------------------------------------------------------

print("\n" + "="*100)
print("TRAINING COMPLETE")
print("="*100)
print(f"Total Episodes: {episodes}")
print(f"Best Reward: {max(reward_history):.0f}")
print(f"Average Reward: {np.mean(reward_history):.0f}")
print(f"Final Average Reward (last 20): {np.mean(reward_history[-20:]):.0f}")
print(f"Average Loss: {np.mean(loss_history):.6f}")
print(f"Final Average Loss (last 20): {np.mean(loss_history[-20:]):.6f}")
print(f"Total Training Steps: {sum(training_steps_per_episode)}")
print(f"Average Training Steps per Episode: {np.mean(training_steps_per_episode):.2f}")
print(f"Model saved to: ./saved_models/dqn_model_final.weights.h5")
print("="*100)