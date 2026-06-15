# Step 4: DQN Training Agent

## Overview

Train the neural network to play Tetris by letting it experience games, storing those experiences, and learning from them. This is where the agent actually improves.

## Running Step 4

### Basic Training

```bash
python code/step4_train_dqn.py --num_epochs 300
```

### With Visualization

```bash
python code/step4_train_dqn.py --num_epochs 300 --render
```

Watch the agent play in real-time while training (slows down training)

### With Experiment Tracking

```bash
pip install wandb
python code/step4_train_dqn.py --num_epochs 300 --wandb
```

Track metrics on Weights & Biases dashboard

### Custom Hyperparameters

```bash
python code/step4_train_dqn.py --num_epochs 500 --lr 0.0005 --batch_size 256
```

## Training Workflow

### High-Level Overview

```
For each episode:
  1. Play one complete game
     - Agent makes decisions
     - Experiences stored in memory
  2. Sample batch from memory
  3. Compute loss using Bellman equation
  4. Update network weights
  5. Print progress
  6. Periodically: update target network, save model
```

### Detailed Step-by-Step

#### 1. Episode: Play One Game

```python
game.reset()
state = get_state_features()

while not game.game_over:
    # Choose action: explore vs exploit
    action = select_action()  # Uses epsilon-greedy
    
    # Execute action
    reward, done, next_state = game.step(action)
    
    # Store experience
    memory.append((state, reward, next_state, done))
    
    # Update state for next iteration
    state = next_state
```

#### 2. Sample Batch

```python
# Random batch of past experiences
batch_size = 512
batch = sample(memory, batch_size)
# batch = [(state1, reward1, next_state1, done1),
#          (state2, reward2, next_state2, done2),
#          ...]
```

#### 3. Compute Target Q-Values

Use **Bellman equation**:

```
Q_target = reward + γ × max_Q(next_state)  [if not game over]
Q_target = reward                           [if game over]
```

In code:
```python
q_target = reward + (1 - done) * gamma * max_q_next_state
```

#### 4. Compute Loss

```python
q_predicted = model(state)
loss = (q_target - q_predicted)²
```

#### 5. Update Network

```python
optimizer.zero_grad()  # Clear old gradients
loss.backward()         # Compute new gradients
optimizer.step()        # Update weights
```

#### 6. Repeat

Repeat steps 1-5 for hundreds of episodes. Network gradually improves.

## Key Concepts

### Epsilon-Greedy Strategy (Exploration vs Exploitation)

**Problem**: If agent only does best action, it gets stuck in local optimum

**Solution**: Sometimes choose random action to explore new strategies

**ε-greedy policy**:
- Probability ε: take random action (explore)
- Probability 1-ε: take best action from network (exploit)

**Epsilon decay over time**:
```
Episodes:  0 → 1000 → 2000 → 3000
ε:        1.0 → 0.5 → 0.001 → 0.001

Early game:  High ε, lots of exploration
Mid game:    Decaying ε, balance exploration/exploitation
Late game:   Low ε, mostly exploitation (use learned strategy)
```

**Impact**:
- Too high ε late: Agent never converges
- Too low ε early: Agent gets stuck in bad strategy
- Decay too fast: Incomplete exploration
- Decay too slow: Wastes time exploring

### Replay Buffer (Experience Memory)

**Problem**: Sequential game states are highly correlated. Training on consecutive steps biases learning.

**Solution**: Store all experiences and sample random batches

```python
memory = deque(maxlen=30000)

# Each step, add experience
memory.append((state, reward, next_state, done))

# When training, sample random batch
batch = random.sample(memory, batch_size=512)
# Batch contains experiences from different times, breaks correlation
```

**Why it helps**:
- Removes temporal correlation between samples
- Uses past experiences more efficiently
- Reduces variance in gradient estimates
- Enables data reuse

### Target Network

**Problem**: Both Q_predicted and Q_target come from the same network that's being updated. This creates instability (moving target).

**Solution**: Use two networks:
- **Q-network**: Updated every step (current weights)
- **Target network**: Copy of Q-network, updated every 100 steps

```python
q_net = DeepQNetwork()      # Updated frequently
target_net = DeepQNetwork() # Updated less frequently

# During training
q_predicted = q_net(state)  # Current network
q_target = reward + gamma * target_net(next_state)  # Stable target

# Every 100 episodes
target_net.load_state_dict(q_net.state_dict())  # Copy weights
```

**Impact**:
- Stabilizes training (target doesn't change each step)
- Reduces oscillations in loss
- Allows longer training without divergence

### Bellman Equation

The fundamental equation of Q-learning:

```
Q(s, a) = E[r + γ × max_Q(s', a')]
```

Interpretation:
- **Q(s, a)**: Expected future reward from state s taking action a
- **r**: Immediate reward from this action
- **γ** (gamma): Discount factor (0.99)
  - How much we value future rewards
  - 0.99: value future almost as much as present
  - 0.9: value future less than present
- **max_Q(s', a')**: Best expected future reward from next state

**Training goal**: Make network outputs satisfy this equation

## Hyperparameters Explained

### Learning Rate (`--lr`)

Default: `0.001`

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Effect**:
- How big each weight update is
- `lr = 0.001`: conservative updates
- `lr = 0.01`: aggressive updates

**Too high**:
```
Loss jumps around, diverges
   │     ╱╲  ╱╲  ╱╲
   │    ╱  ╲╱  ╲╱
   └──────────────
```

**Too low**:
```
Loss decreases slowly, takes forever
   │╲╲╲╲╲╲╲╲╲╲╲
   │ ╲
   │  ╲
   └────────────
```

**Just right**:
```
Loss decreases smoothly, converges
   │╲
   │ ╲╲
   │  ╲╲╲
   └────╲╲╲─
```

**Tips**:
- Start with 0.001
- If loss diverges, reduce to 0.0005
- If learning is too slow, try 0.002

### Batch Size (`--batch_size`)

Default: `512`

```python
batch = sample(memory, batch_size=512)
```

**Effect**:
- How many experiences in each training step
- Larger: more stable estimates
- Smaller: noisier but faster updates

**Larger batch (1024)**:
- Stable loss curves
- Slower training (fewer updates per epoch)
- Better convergence

**Smaller batch (64)**:
- Noisy loss curves (lots of variance)
- Faster training (more updates per epoch)
- Can diverge if too small

**Tips**:
- 512 is standard for DQN
- If learning is unstable, increase to 1024
- If training is slow, reduce to 256

### Gamma (Discount Factor) (`--gamma`)

Default: `0.99`

```python
q_target = reward + gamma * q_next
```

**Effect**:
- How much agent values future rewards

**γ = 0.99** (default):
- Values future highly
- Agent plans long-term
- Leads to better play

**γ = 0.95**:
- Values future less
- Agent more myopic
- Plays more greedily

**γ = 0.9**:
- Values future much less
- Agent ignores long-term consequences
- Poor strategy

**Tips**:
- 0.99 works well for Tetris
- Don't change unless you have reason to

### Epsilon Decay (`--decay_epochs`)

Default: `2000`

```python
epsilon = final_eps + max(decay_epochs - episode, 0) * (initial_eps - final_eps) / decay_epochs
```

**Effect**:
- How many episodes to transition from explore to exploit

**Decay over 2000 episodes**:
- Episodes 0-1000: ε goes from 1.0 → 0.5
- Episodes 1000-2000: ε goes from 0.5 → 0.001
- Episodes 2000+: ε stays at 0.001

**Decay over 500 episodes**:
- Fast transition to exploitation
- Risk: incomplete exploration
- Good if you notice agent learns quickly

**Decay over 5000 episodes**:
- Slow transition
- More thorough exploration
- Good if learning is plateauing

**Tips**:
- Start with 2000
- If agent improves then plateaus, increase to 3000-5000
- If agent converges quickly, reduce to 1000-1500

## Expected Training Curves

### Score Over Episodes

```
Score
  │     ╱╱╱
  │    ╱╱
  │   ╱╱
  │  ╱╱
  │ ╱
  │╱━━━━━━ Episodes
 0│
```

- Episodes 0-100: Mostly random, scores ~10-50
- Episodes 100-300: Improving, scores ~50-200
- Episodes 300+: Plateaus or improves slowly, scores ~100-300+

### Loss Over Episodes

```
Loss
  │
  │╲
  │ ╲╲╲╲
  │     ╲╲╲╲━━━━━━ Episodes
  │
```

- Early: Loss decreases rapidly (learning patterns)
- Mid: Loss decreases slowly (fine-tuning)
- Late: Loss stabilizes (converged)

### Epsilon Decay

```
Epsilon
  1.0│╲
      │ ╲╲
  0.5│   ╲╲╲╲
      │       ╲╲╲━━━ Episodes
 0.001│
```

- Linear decay from 1.0 to 0.001
- Reflects increasing exploitation over time

## Training Tips

### 1. Start Small

```bash
python code/step4_train_dqn.py --num_epochs 100
```

- Quick iteration
- Test if setup works
- Check output format

### 2. Monitor Loss

Look for patterns in loss output:
- Decreasing: Good! Training is working
- Constant: Agent isn't learning (check lr, batch_size)
- Increasing: Something is wrong (divergence)
- NaN: Exploding gradients (reduce lr)

### 3. Check Score Improvement

- Early episodes: Scores should be terrible (0-100)
- After 200 episodes: Scores should be 100-300
- After 500 episodes: Scores should be 200-500

If scores aren't improving, try:
- Increase epsilon decay (more exploration)
- Lower learning rate (more stable)
- Larger batch size (more stable)

### 4. Save Frequently

```bash
python code/step4_train_dqn.py --num_epochs 1000 --save_interval 50
```

Save every 50 episodes, so you can:
- Test intermediate models
- Recover if training diverges
- Analyze learning progress

### 5. Use Wandb for Tracking

```bash
pip install wandb
python code/step4_train_dqn.py --num_epochs 500 --wandb
```

Automatically logs:
- Game metrics (score, lines, pieces)
- Model metrics (loss, epsilon)
- Creates graphs for analysis

## Troubleshooting

### Loss is NaN

**Cause**: Exploding gradients

**Fix**:
```bash
python code/step4_train_dqn.py --lr 0.0001  # Lower learning rate
```

### Agent Not Learning

**Cause**: Learning rate too low, not enough exploration

**Fix**:
```bash
python code/step4_train_dqn.py --lr 0.005 --decay_epochs 3000
```

### Training Too Slow

**Cause**: Batch size too large, learning rate too low

**Fix**:
```bash
python code/step4_train_dqn.py --batch_size 256 --lr 0.002
```

### Model Overfits (high train score, low test score)

**Cause**: Network too large, training too long

**Fix**:
```python
# In step3_neural_network.py, reduce layer sizes
model = DeepQNetwork(hidden1_size=32, hidden2_size=16)
```

## Advanced: What's Happening Inside

### Memory Structure

```python
memory = deque(maxlen=30000)
# [(state, reward, next_state, done), ...]
# Example:
# ([5, 2, 10, 15], 101, [5, 1, 8, 16], False)
# ([5, 1, 8, 16], -2.5, [5, 0, 10, 18], False)
# ([5, 0, 10, 18], -5, [0, 0, 0, 0], True)
```

### Batch Processing

```python
batch = sample(memory, 512)
states = [s for s, r, ns, d in batch]  # 512 states
rewards = [r for s, r, ns, d in batch]  # 512 rewards
next_states = [ns for s, r, ns, d in batch]  # 512 next states
dones = [d for s, r, ns, d in batch]  # 512 done flags

# Convert to tensors
s_tensor = torch.FloatTensor(states)      # Shape: (512, 4)
q_pred = model(s_tensor)                  # Shape: (512, 1)
```

### Target Computation

```python
# Bellman update for whole batch
q_next = target_network(next_states_tensor)  # Shape: (512, 1)
max_q_next = torch.max(q_next, dim=1)[0]     # Max per state: (512,)

q_target = rewards_tensor + (1 - dones_tensor) * gamma * max_q_next
```

## Key Takeaways

1. **Epsilon-greedy balances exploration and exploitation**: Start exploring, gradually exploit

2. **Replay buffer breaks correlation**: Sample random batches for stable learning

3. **Target network provides stable targets**: Update less frequently than main network

4. **Bellman equation drives learning**: Network learns to satisfy this equation

5. **Hyperparameters matter**: Learning rate, batch size, gamma all affect training

6. **Monitor loss and score**: Track learning progress, adjust if needed

## Next Steps

- **Step 5**: Test the trained model
- Experiment with hyperparameters
- Try different network architectures
- Implement advanced techniques (double DQN, dueling DQN)
