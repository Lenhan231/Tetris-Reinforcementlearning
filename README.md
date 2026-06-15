# 🎮 Tetris Deep Q-Learning: From Scratch to Playing Agent

A complete hands-on guide to building a neural network that learns to play Tetris using Deep Reinforcement Learning (DQN). This project breaks down complex RL concepts into digestible steps with working code.

## 🎯 Project Overview

This project teaches you:
- **Game Logic**: Build a complete Tetris engine with proper physics and mechanics
- **State Representation**: Extract meaningful features from game states
- **Neural Networks**: Design and train a deep Q-network
- **Reinforcement Learning**: Implement DQN training with experience replay
- **Model Evaluation**: Test and visualize your trained agent playing

## 📁 Project Structure

```
tetris_from_scratch/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── code/                        # Implementation code
│   ├── step1_tetris_basic.py       # Tetris game engine
│   ├── step3_neural_network.py     # Deep Q-Network architecture
│   ├── step4_train_dqn.py          # Training loop & DQN agent
│   └── step5_test_model.py         # Model testing & evaluation
└── models/                      # Trained model checkpoints
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Requires: PyTorch, NumPy, OpenCV (for rendering)

### 2. Understand the Game (Step 1)

```bash
python code/step1_tetris_basic.py
```

This runs a quick demo of the Tetris game engine. You'll see:
- Board representation (20×10 grid)
- Piece mechanics (rotation, collision, placement)
- Line clearing and scoring
- State features extraction

**Learn**: How the game works and what data the AI will see.

### 3. Study the Neural Network (Step 3)

```bash
python code/step3_neural_network.py
```

This demonstrates the architecture of the Deep Q-Network:
- Input: 4 features (lines_cleared, holes, bumpiness, height)
- Hidden layers: 64 → 32 neurons
- Output: 1 Q-value (expected future reward)
- Training example using Bellman equation

**Learn**: How neural networks estimate Q-values and improve through training.

### 4. Train Your Agent (Step 4)

```bash
python code/step4_train_dqn.py --num_epochs 300
```

Optional flags:
- `--num_epochs 500` - Train for longer (default: 100)
- `--batch_size 256` - Smaller batches (default: 512)
- `--lr 0.0005` - Lower learning rate (default: 0.001)
- `--render` - Watch the agent train (slows down training)

**What happens during training**:
1. Agent plays Tetris games with random and greedy actions (epsilon-greedy)
2. Experiences get stored in a replay buffer
3. Batches of experiences are used to train the neural network
4. Model improves after each update
5. Model checkpoints saved every 100 episodes

**What to expect**:
- Episode 1-100: Random play, terrible scores
- Episode 100-500: Agent starts learning, scores improve
- Episode 500+: Consistent gameplay, good line clears

### 5. Test Your Model (Step 5)

```bash
python code/step5_test_model.py --model_path models/tetris_final.pth --num_games 50
```

This evaluates the trained model:
- Plays 50 games without training
- Shows statistics (average score, pieces, lines, steps)
- Displays performance rating

**Optional**: Use `--infinite` for interactive mode (pause, rewind, restart).

## 📚 Detailed Step Guide

### Step 1: Tetris Game Engine (`step1_tetris_basic.py`)

**Purpose**: Build a working Tetris game that the AI will play.

**Key components**:
- **Board**: 20×10 grid with piece placement
- **Pieces**: 7 tetrominos (O, T, S, Z, I, L, J) with 4 rotation states
- **Physics**: Collision detection, gravity, line clearing
- **State Representation**: Compute 4 features for the AI
  - `lines_cleared`: Total lines cleared so far
  - `holes`: Unfilled gaps under placed blocks
  - `bumpiness`: Height variation between columns
  - `height`: Total height of the stack

**API**:
```python
game = TetrisGame(height=20, width=10)
state = game.reset()  # Returns (lines, holes, bumpiness, height)

next_states = game.get_next_states()  # Dict of all possible moves
action = (x_position, num_rotations)
reward, done, state = game.step(action)
```

### Step 3: Deep Q-Network (`step3_neural_network.py`)

**Purpose**: Build a neural network that predicts Q-values.

**Architecture**:
```
Input (4) → Dense(64) + ReLU → Dense(32) + ReLU → Output (1)
```

**Key concepts**:
- **Input**: Game state features (4 values)
- **Output**: Single Q-value (expected future reward)
- **ReLU**: Adds non-linearity so the network can learn complex patterns
- **Why this size**: 64→32 neurons is small enough to train quickly but large enough to learn patterns

**Training intuition**:
- Target Q-value (Bellman): `Q_target = reward + γ × max_Q(next_state)`
- Loss: `(Q_predicted - Q_target)²`
- Update: Use gradient descent to minimize loss

### Step 4: DQN Training (`step4_train_dqn.py`)

**Purpose**: Train the agent to play Tetris better over time.

**Key mechanisms**:

1. **Epsilon-Greedy Strategy**:
   - Early training (ε=1.0): 100% random actions (explore)
   - Late training (ε≈0.001): 99% best actions from network (exploit)
   - Smooth decay over 2000 episodes

2. **Replay Buffer**:
   - Stores up to 30,000 past experiences: (state, reward, next_state, done)
   - Sampling random batches breaks correlation between consecutive samples
   - Helps prevent overfitting to recent games

3. **Target Network**:
   - Separate copy of Q-network used to compute targets
   - Updated every 100 episodes
   - Prevents "chasing a moving target" during training

4. **Bellman Update**:
   - Formula: `Q(s,a) = r + γ × max_Q(s', a')`
   - If game ends: `Q(s,a) = r`
   - Network learns to satisfy this equation

**Expected training curve**:
- Score increases gradually
- Loss decreases then stabilizes
- Agent learns to clear lines consistently

### Step 5: Model Testing (`step5_test_model.py`)

**Purpose**: Evaluate a trained model on fresh games.

**Features**:
- Plays multiple games without training
- Shows statistics: mean, std, min, max
- Performance rating based on lines cleared
- Optional visualization with pygame

## 🎓 Key RL Concepts

### Q-Learning Basics

**Q-Value**: Expected cumulative future reward from state s when taking action a
```
Q(s, a) = immediate_reward + γ × expected_future_rewards
```

**Bellman Equation**: 
```
Q(s, a) = r + γ × max_Q(s', a')
```
Where:
- `r`: Immediate reward
- `γ` (gamma): Discount factor (0.99) - how much we value future rewards
- `s'`: Next state
- `a'`: Best action in next state

### Deep Q-Network (DQN)

Instead of storing Q-values in a table (impossible with high-dimensional states), use a neural network:
- Input: State description
- Output: Predicted Q-value
- Training: Minimize loss between predicted and target Q-values

### Experience Replay

Problem: Sequential game states are highly correlated (sample batches would be biased)

Solution: Store experiences in a buffer and sample random batches for training
- Breaks correlation
- Improves data efficiency
- More stable training

## 💡 Hyperparameters and Tuning

### Training parameters (in `step4_train_dqn.py`):

- **Learning rate** (`--lr`): How quickly network updates (default: 0.001)
  - Higher → faster learning but less stable
  - Lower → slower learning but more stable

- **Batch size** (`--batch_size`): Number of experiences per training step (default: 512)
  - Larger → more stable but slower
  - Smaller → noisier but faster

- **Gamma** (`--gamma`): Discount factor (default: 0.99)
  - Higher → values future rewards more
  - Lower → focuses on immediate rewards

- **Epsilon decay** (`--decay_epochs`): How many episodes to decay exploration (default: 2000)
  - Longer → more exploration
  - Shorter → quicker transition to exploitation

### Game rewards:

The agent receives rewards for:
- **Positive**: Clearing lines (1 line = 101 points, 4 lines = 401 points)
- **Negative**: Building tall stacks, creating holes, bumpy surface
- **Penalty**: Game over (-10)

## 🔍 Troubleshooting

### Training is slow
- Reduce `--num_epochs` for faster iterations
- Disable `--render` flag (visualization is slow)
- Use GPU if available (automatically detected)

### Model not improving
- Check learning rate isn't too high (try 0.0005)
- Increase `--num_epochs` (training needs time)
- Verify loss is decreasing in training logs

### Testing shows poor performance
- Train for more episodes (model might not be converged)
- Check model file exists at specified path
- Try lower `--speed` value in testing for visual inspection

## 📊 Performance Expectations

**After 100 episodes**: Random gameplay, ~5-20 lines per game

**After 300 episodes**: Learning visible, ~20-50 lines per game

**After 500+ episodes**: Consistent play, ~50-150 lines per game

**Excellent model**: 100+ lines average per game

## 🛠️ Next Steps & Experiments

Once the basic model works, try:

1. **Architecture changes**: Modify hidden layer sizes in `DeepQNetwork`
2. **Reward shaping**: Adjust the reward penalties in `step1_tetris_basic.py`
3. **Different features**: Add more state features in `_get_state_features()`
4. **Dueling DQN**: Separate value and advantage streams
5. **Prioritized Experience Replay**: Sample important experiences more often

## 📖 References

- [Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book-2nd.html)
- [DQN Paper](https://www.nature.com/articles/nature14236)
- [PyTorch Documentation](https://pytorch.org/)

## 📝 License

Educational project. Use freely for learning.

---

**Happy Learning! 🚀**

Start with Step 1 to understand the game, then progress through each step in order.
