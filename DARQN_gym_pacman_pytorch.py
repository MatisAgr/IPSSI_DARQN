# version 2 pour la curiosité avec pytorch à la place de tensorflow
# utilisation du gpu et des paramètre plus élevés pour l'entraînement

import numpy as np
import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from collections import deque
import time
import os
import signal
import sys
import argparse
import json
from torchvision import transforms

# ------------------------------------------------------------------------
# GPU CONFIGURATION
# ------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No GPU detected, using CPU")

# Set number of threads for CPU operations
torch.set_num_threads(14)

gym.register_envs(ale_py)

# ------------------------------------------------------------------------
# HYPERPARAMETERS
# ------------------------------------------------------------------------
# Optimized hyperparameters for stable training:
# - Lower learning rate (0.0001) prevents oscillations
# - Huber loss reduces sensitivity to outliers
# - Reward clipping normalizes learning signals
# - Stricter gradient clipping (1.0) prevents exploding gradients

env_name = "ALE/Pacman-v5"
learning_rate = 0.0001  # Reduced from 0.001 for more stable training
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
gamma = 0.99
batch_size = 128
memory_size = 50000
episodes = 500
update_target_frequency = 5
render_mode = None  # Set to "human" to see the game, None for faster training
loss_function = 'huber'  # Changed from 'mse' to 'huber' for robustness
train_interval = 4
frame_stack = 4
reward_clip = True  # Enable reward clipping for stability
gradient_clip_norm = 1.0  # Reduced from 10.0 for better stability

# ------------------------------------------------------------------------
# COMMAND LINE ARGUMENTS
# ------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Train DARQN agent on Pacman')
parser.add_argument('--resume', type=str, default=None,
                    help='Resume training from checkpoint (e.g., 180 to load episode_180)')
parser.add_argument('--episodes', type=int, default=episodes,
                    help='Number of episodes to train')
args = parser.parse_args()

if args.episodes != episodes:
    episodes = args.episodes

resume_from_episode = 0
resume_checkpoint = args.resume

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
    obs_resized = torch.tensor(obs_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    obs_resized = F.interpolate(obs_resized, size=(84, 84), mode='bilinear', align_corners=False)
    return obs_resized.squeeze().numpy() / 255.0

def get_stacked_frames(frame):
    """Stack frames for temporal context"""
    frame_buffer.append(preprocess_observation(frame))
    stacked = np.stack(list(frame_buffer), axis=-1)
    return stacked

def initialize_frame_buffer(initial_obs):
    """Initialize frame buffer with repeated frames"""
    frame_buffer.clear()
    processed = preprocess_observation(initial_obs)
    for _ in range(frame_stack):
        frame_buffer.append(processed)
    return np.stack(list(frame_buffer), axis=-1)

state_shape = (84, 84, frame_stack)

# ------------------------------------------------------------------------

os.makedirs('./saved_models', exist_ok=True)
os.makedirs('./logs', exist_ok=True)

# TensorBoard writer for PyTorch
writer = SummaryWriter(log_dir='./logs')

# ATTENTION MECHANISM (CBAM)
# ------------------------------------------------------------------------

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    
    Applies channel-wise and spatial attention to learned features.
    - Channel Attention: Which feature maps are important?
    - Spatial Attention: Which spatial locations are important?
    """
    def __init__(self, channels, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        
        hidden = max(channels // ratio, 1)
        
        # Channel attention: FC layers with bottleneck
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        
        # Spatial attention: Conv layer to learn spatial patterns
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        # Channel Attention: Learn which feature maps are important
        batch, channels, height, width = x.size()
        
        # Global average pooling and max pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch, channels)
        
        # Pass through FC layers
        avg_pool = self.fc2(F.relu(self.fc1(avg_pool)))
        max_pool = self.fc2(F.relu(self.fc1(max_pool)))
        
        # Combine and apply sigmoid
        channel_attn = torch.sigmoid(avg_pool + max_pool).view(batch, channels, 1, 1)
        
        # Apply channel attention to features
        x = x * channel_attn
        
        # Spatial Attention: Learn which spatial locations are important
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_attn = torch.sigmoid(self.spatial_conv(spatial))
        
        # Apply spatial attention
        return x * spatial_attn

# ------------------------------------------------------------------------
# DARQN MODEL CREATION
# ------------------------------------------------------------------------

class DARQNModel(nn.Module):
    """DARQN (Dueling Attention Recurrent Q-Network) model"""
    def __init__(self, state_shape, action_space):
        super(DARQNModel, self).__init__()
        
        # -------- CNN FEATURE EXTRACTION BRANCH --------
        self.conv1 = nn.Conv2d(state_shape[2], 32, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # -------- ATTENTION MECHANISM (CBAM) --------
        self.attention = CBAM(channels=128, ratio=8, kernel_size=7)
        
        # Calculate the size after convolutions
        # Input: (84, 84, 4)
        # After conv1 (stride=4): (21, 21, 32)
        # After conv2 (stride=2): (11, 11, 64)
        # After conv3 (stride=1): (11, 11, 128)
        self.conv_output_size = 11 * 11 * 128
        
        # -------- RECURRENT PROCESSING (TEMPORAL CONTEXT) --------
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, 
                            batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, 
                            batch_first=True, dropout=0.2)
        
        # Use LazyLinear to automatically infer input size from first forward pass
        # This handles the dynamic merge size (CNN flatten + LSTM output)
        
        # -------- DENSE PROCESSING LAYERS --------
        self.dense1 = nn.LazyLinear(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.dense2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        
        # -------- DUELING ARCHITECTURE --------
        # Value Stream: estimates state value V(s)
        self.value_dense = nn.Linear(256, 128)
        self.value_output = nn.Linear(128, 1)
        
        # Advantage Stream: estimates action advantages A(s,a)
        self.advantage_dense = nn.Linear(256, 128)
        self.advantage_output = nn.Linear(128, action_space)
    
    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Attention mechanism
        x = self.attention(x)
        
        # Get actual spatial dimensions after convolutions
        batch_size, channels, height, width = x.size()
        seq_len = height * width
        
        # Flatten for dense layers
        flat = x.reshape(batch_size, -1)
        
        # Reshape for LSTM: (batch, seq_len, channels)
        # Each spatial position becomes a timestep with all channel features
        lstm_input = x.permute(0, 2, 3, 1).contiguous()  # (batch, H, W, C)
        lstm_input = lstm_input.reshape(batch_size, seq_len, channels)
        
        # LSTM temporal processing
        lstm_out, _ = self.lstm1(lstm_input)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = lstm_out[:, -1, :]  # Take last timestep
        
        # Merge CNN and LSTM features
        merge = torch.cat([flat, lstm_out], dim=1)
        
        # Dense processing
        x = F.relu(self.dense1(merge))
        x = self.dropout1(x)
        x = F.relu(self.dense2(x))
        x = self.dropout2(x)
        
        # Dueling architecture
        # Value stream
        value = F.relu(self.value_dense(x))
        value = self.value_output(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_dense(x))
        advantage = self.advantage_output(advantage)
        
        # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values

q_model = DARQNModel(state_shape, action_space).to(device)
target_model = DARQNModel(state_shape, action_space).to(device)
target_model.load_state_dict(q_model.state_dict())
target_model.eval()  # Set target model to evaluation mode

# Optimizer and loss function
optimizer = optim.Adam(q_model.parameters(), lr=learning_rate)
# Use Huber Loss (SmoothL1Loss) for robustness against outliers
criterion = nn.SmoothL1Loss() if loss_function == 'huber' else nn.MSELoss()

# Print model summary
print("\n" + "="*100)
print("DARQN MODEL ARCHITECTURE")
print("="*100)
print(q_model)
print("="*100 + "\n")

# Load checkpoint if resuming training
if resume_checkpoint:
    checkpoint_path = f'./saved_models/darqn_model_episode_{resume_checkpoint}.pth'
    if os.path.exists(checkpoint_path):
        print(f"\n{'='*100}")
        print(f"Loading checkpoint from episode {resume_checkpoint}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        q_model.load_state_dict(checkpoint['q_model_state_dict'])
        target_model.load_state_dict(checkpoint['target_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_from_episode = checkpoint['episode']
        print(f"Weights loaded successfully!")
        print(f"Resuming training from episode {resume_from_episode + 1}...")
        print(f"{'='*100}\n")
    else:
        print(f"\n{'='*100}")
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"Available checkpoints in ./saved_models/:")
        checkpoint_files = [f for f in os.listdir('./saved_models/') if 'episode_' in f and f.endswith('.pth')]
        for f in sorted(checkpoint_files):
            print(f"  - {f}")
        print(f"{'='*100}\n")
        sys.exit(1)

# ------------------------------------------------------------------------

memory = deque(maxlen=memory_size)

def store_transition(state, action, reward, next_state, done):
    # Clip rewards for stability if enabled
    if reward_clip:
        reward = np.clip(reward, -1, 1)
    memory.append((state, action, reward, next_state, done))

def sample_batch():
    """Sample a batch from memory"""
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Convert to numpy arrays - states are already (84, 84, 4)
    # Just stack them to create batch dimension
    states_array = np.array(states, dtype=np.float32)
    next_states_array = np.array(next_states, dtype=np.float32)
    
    return (
        states_array,
        np.array(list(actions), dtype=np.int32),
        np.array(list(rewards), dtype=np.float32),
        next_states_array,
        np.array(list(dones), dtype=np.float32)
    )

def epsilon_greedy_policy(state, epsilon):
    """Epsilon-greedy action selection"""
    if np.random.random() < epsilon:
        return np.random.choice(action_space)
    else:
        q_model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            # Permute to match PyTorch format: (batch, channels, height, width)
            state_tensor = state_tensor.permute(0, 3, 1, 2)
            q_values = q_model(state_tensor)
        q_model.train()
        return q_values.cpu().numpy().argmax()

# ------------------------------------------------------------------------
# TRAINING STEP
# ------------------------------------------------------------------------

def train_step():
    """Perform one training step"""
    if len(memory) < batch_size:
        return None
    
    states, actions, rewards, next_states, dones = sample_batch()
    
    # Convert to PyTorch tensors and move to device
    # States are already in format (batch, H, W, C) = (batch, 84, 84, 4)
    # Need to permute to PyTorch format: (batch, C, H, W) = (batch, 4, 84, 84)
    states = torch.FloatTensor(states).permute(0, 3, 1, 2).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).permute(0, 3, 1, 2).to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    # Compute current Q-values
    q_model.train()
    current_q_values = q_model(states)
    current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute target Q-values using target network
    target_model.eval()
    with torch.no_grad():
        next_q_values = target_model(next_states)
        max_next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
    
    # Compute loss
    loss = criterion(current_q_values, target_q_values)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients (stricter)
    torch.nn.utils.clip_grad_norm_(q_model.parameters(), max_norm=gradient_clip_norm)
    
    optimizer.step()
    
    return loss.item()


# ------------------------------------------------------------------------
# MAIN TRAINING SCRIPT
# ------------------------------------------------------------------------

if __name__ == '__main__':
    
    reward_history = []
    loss_history = []
    training_steps_per_episode = []

    # Load previous metrics if resuming
    if resume_from_episode > 0:
        metrics_file = './metrics/training_metrics.json'
        if os.path.exists(metrics_file):
            print(f"Loading previous metrics from {metrics_file}...")
            with open(metrics_file, 'r') as f:
                previous_metrics = json.load(f)
            reward_history = previous_metrics.get('reward_history', [])[:resume_from_episode]
            loss_history = previous_metrics.get('loss_history', [])[:resume_from_episode]
            training_steps_per_episode = previous_metrics.get('training_steps_per_episode', [])[:resume_from_episode]
            
            # Adjust epsilon based on episodes trained
            epsilon = max(epsilon_min, epsilon * (epsilon_decay ** resume_from_episode))
            
            print(f"Loaded {resume_from_episode} episodes of previous training")
            print(f"- Previous best reward: {max(reward_history):.0f}")
            print(f"- Previous avg reward: {np.mean(reward_history):.0f}")
            print(f"- Current epsilon: {epsilon:.4f}\n")

    # SIGNAL HANDLER FOR GRACEFUL SHUTDOWN (CTRL+C)
    # ------------------------------------------------------------------------

    interrupted = False

    def signal_handler(sig, frame):
        """Handle CTRL+C to save model before exit"""
        global interrupted, episode, q_model, target_model, optimizer, epsilon, reward_history, loss_history, training_steps_per_episode
        interrupted = True
        print("\n" + "="*100)
        print("INTERRUPT SIGNAL RECEIVED - SAVING MODEL AND METRICS")
        print("="*100)
        
        # Save complete model checkpoint
        checkpoint_path = f'./saved_models/darqn_model_interrupt_ep{episode+1}.pth'
        checkpoint = {
            'episode': episode + 1,
            'q_model_state_dict': q_model.state_dict(),
            'target_model_state_dict': target_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon,
            'reward_history': reward_history,
            'loss_history': loss_history
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved to: {checkpoint_path}")
        
        # Save metrics
        os.makedirs('./metrics', exist_ok=True)
        metrics_data = {
            'episodes': list(range(1, episode + 2)),
            'reward_history': [float(r) for r in reward_history],
            'loss_history': [float(l) for l in loss_history],
            'training_steps_per_episode': [int(s) for s in training_steps_per_episode],
        }
        with open('./metrics/training_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Metrics saved to: ./metrics/training_metrics.json")
        
        # Close writer
        writer.close()
        
        # Print summary
        print(f"\nTraining interrupted at episode {episode+1}")
        print(f"Best Reward: {max(reward_history):.0f}")
        print(f"Average Reward: {np.mean(reward_history):.0f}")
        print(f"Average Loss: {np.mean(loss_history):.6f}")
        print("="*100 + "\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # TRAINING LOOP
    # ------------------------------------------------------------------------

    for episode in range(resume_from_episode, episodes):
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
            target_model.load_state_dict(q_model.state_dict())
        
        # Store metrics
        reward_history.append(total_reward)
        training_steps_per_episode.append(training_steps)
        
        # Calculate episode statistics
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        min_loss = np.min(episode_losses) if episode_losses else 0
        max_loss = np.max(episode_losses) if episode_losses else 0
        loss_history.append(avg_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Reward/Episode', total_reward, episode+1)
        writer.add_scalar('Loss/Episode', avg_loss, episode+1)
        writer.add_scalar('Epsilon', epsilon, episode+1)
        writer.add_scalar('Steps', steps, episode+1)
        
        # Print episode info
        print(f"Episode: {episode+1}/{episodes} | Reward: {total_reward:.0f} | Eps: {epsilon:.3f} | Time: {elapsed_time:.2f}s | Steps: {steps} | Memory: {len(memory)}")
        print(f"  └─ Loss -> Avg: {avg_loss:.6f} | Min: {min_loss:.6f} | Max: {max_loss:.6f} | Trainings: {training_steps}")

        # Save model checkpoint every 20 episodes
        if (episode + 1) % 20 == 0:
            checkpoint = {
                'episode': episode + 1,
                'q_model_state_dict': q_model.state_dict(),
                'target_model_state_dict': target_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epsilon': epsilon,
                'reward_history': reward_history,
                'loss_history': loss_history
            }
            torch.save(checkpoint, f'./saved_models/darqn_model_episode_{episode+1}.pth')
            print(f"\n{'='*100}")
            print(f"Model checkpoint saved at episode {episode+1}")
            print(f"Reward -> Best: {max(reward_history):.0f} | Avg (last 20): {np.mean(reward_history[-20:]):.0f}")
            print(f"Loss   -> Avg (last 20): {np.mean(loss_history[-20:]):.6f}")
            print(f"{'='*100}\n")

    env.close()
    writer.close()

    # Save final model
    torch.save({
        'episode': episodes,
        'q_model_state_dict': q_model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'reward_history': reward_history,
        'loss_history': loss_history
    }, './saved_models/darqn_model_final.pth')
    print("Model saved to ./saved_models/darqn_model_final.pth")

    # Export metrics to JSON for visualization
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
    print(f"Model saved to: ./saved_models/darqn_model_final.pth")
    print("="*100)