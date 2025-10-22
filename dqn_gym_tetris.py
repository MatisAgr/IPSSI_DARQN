import numpy as np
from keras import layers
import gymnasium as gym # maj vers gymnasium
import ale_py # Support pour ALE/Tetris
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

# Actions Tetris:
# 0: NON (pas d'action)
# 1: FEU (rotation)
# 2: DROITE (vers la droite)
# 3: GAUCHE (vers la gauche)
# 4: VERS LE BAS (descendre)

# ------------------------------------------------------------------------


env_name = "ALE/Tetris-v5"      # env name ALE
learning_rate = 0.001           # learning rate
epsilon = 1.0                   # exploration rate
epsilon_min = 0.2               # min exploration rate
epsilon_decay = 0.995           # decay rate
gamma = 0.95                    # discount factor
batch_size = 64                 # batch size
memory_size = 10000             # replay memory size
episodes = 100                  # number of episodes
update_target_frequency = 10    # update target model every 10 episodes
render_mode = None              # "human" for UI
loss = 'mse'                    # loss function
train_interval = 4              # train every 4 steps


# custom reward vars
one_line_clear_reward = 10.0        # reward for clearing one line
two_lines_clear_reward = 30.0       # reward for clearing two lines (3x bonus)
three_lines_clear_reward = 70.0     # reward for clearing three lines (7x bonus)
four_lines_clear_reward = 150.0     # reward for clearing four lines (15x bonus)

survival_reward = 0.5               # reward per step (encourages staying alive)
game_over_penalty = -50.0           # penalty for game over

positive_reward_multiplier = 5.0    # multiplier for positive original rewards (5x)
negative_reward_multiplier = 2.0    # multiplier for negative original rewards (2x)
score_increase_divisor = 40         # divisor to estimate lines cleared from score

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

def create_q_model():
    model = tf.keras.Sequential(
        [
            layers.Dense(256, activation='relu', input_shape=(state_shape, )),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_shape, activation='linear')
        ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        # jit_compile=True  # XLA compilation for speedup
    )
    return model

q_model = create_q_model()
target_model = create_q_model()
target_model.set_weights(q_model.get_weights())


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
    custom_reward = 0
    
    # penalty if game over
    if done:
        custom_reward += game_over_penalty
        return custom_reward
    
    # extract information if available
    current_score = info.get('score', 0)
    prev_score = prev_info.get('score', 0) if prev_info else 0
    
    # Reward based on score increase
    score_increase = current_score - prev_score
    
    # Detection of completed lines (score usually increases in steps)
    if score_increase > 0:
        # Estimation of number of lines (adjusted according to game scoring)
        lines_cleared = score_increase // score_increase_divisor
        
        if lines_cleared == 1:
            custom_reward += one_line_clear_reward
        elif lines_cleared == 2:
            custom_reward += two_lines_clear_reward
        elif lines_cleared == 3:
            custom_reward += three_lines_clear_reward
        elif lines_cleared >= 4:
            custom_reward += four_lines_clear_reward
        else:
            custom_reward += score_increase  # Small bonus for other actions
    
    # survival reward (encourage staying alive)
    custom_reward += survival_reward
    
    # Bonus for positive original reward (successful placement)
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