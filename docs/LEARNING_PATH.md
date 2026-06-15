# Complete Learning Path: Tetris DQN from Scratch

## 🎯 What You'll Learn

By the end of this project, you'll understand:

- **Game Development**: Build a complete Tetris game with proper physics
- **Reinforcement Learning**: How agents learn through trial and error
- **Neural Networks**: Design and train deep learning models
- **Deep Q-Learning**: Implement the DQN algorithm
- **Model Evaluation**: Test and analyze trained agents

## 📅 Recommended Learning Schedule

### Time Commitment
- **Reading + Understanding**: 4-6 hours
- **Running code + Experimenting**: 6-10 hours
- **Total**: ~10-15 hours (can be spread over 2-3 weeks)

### Suggested Pacing

**Day 1-2: Foundations**
- Read reinforcement learning concepts
- Run Step 1 (understand the game)
- Run Step 3 (understand the network)

**Day 3: Implementation**
- Run Step 4 training (300 episodes, ~30 minutes)
- Monitor training progress
- Observe loss decreasing

**Day 4: Evaluation**
- Run Step 5 (test the model)
- Analyze statistics
- Compare with random baseline

**Day 5: Experimentation**
- Modify hyperparameters
- Train different models
- Compare results

## 📚 Step-by-Step Learning Guide

### Phase 1: Understanding the Game (Days 1-2)

**Goal**: Understand what the AI will learn to play

#### Reading
1. **STEP1_GAME_ENGINE.md**: Read "Game Basics" and "State Features"
2. **README.md**: Read "Project Overview"

#### Hands-On
```bash
# Run the game engine demo
python code/step1_tetris_basic.py
```

**Questions to answer**:
- [ ] What are the 7 tetromino pieces?
- [ ] How are state features (lines, holes, bumpiness, height) calculated?
- [ ] What actions can the agent take?
- [ ] How is reward calculated?

**Experiments**:
- [ ] Modify the demo to play 20 games instead of 5
- [ ] Track average scores
- [ ] Print board states to understand game flow

**Code to try**:
```python
from step1_tetris_basic import TetrisGame

game = TetrisGame()
game.reset()

# Play random game
while not game.game_over:
    states = game.get_next_states()
    action = random.choice(list(states.keys()))
    reward, done, state = game.step(action)
    print(f"Reward: {reward:.2f}, State: {state}")
    
print(f"Final score: {game.score}")
```

### Phase 2: Understanding Neural Networks (Days 2-3)

**Goal**: Understand how neural networks estimate Q-values

#### Reading
1. **STEP3_NEURAL_NETWORK.md**: Read entire document
2. **README.md**: Read "Key RL Concepts"

#### Hands-On
```bash
# Run the neural network demo
python code/step3_neural_network.py
```

**Questions to answer**:
- [ ] What is a Q-value?
- [ ] How many parameters does the network have?
- [ ] What does ReLU activation do?
- [ ] How is the network trained?

**Experiments**:
- [ ] Create a network and predict Q-values for sample states
- [ ] Modify layer sizes and see parameter count change
- [ ] Manually compute network output for one state

**Code to try**:
```python
from step3_neural_network import DeepQNetwork
import torch

model = DeepQNetwork()
model.summary()

# Test prediction
state = torch.FloatTensor([10, 5, 20, 15])
with torch.no_grad():
    q = model(state)
    print(f"Q-value: {q.item():.4f}")

# Try different network sizes
small_model = DeepQNetwork(hidden1_size=32, hidden2_size=16)
print(f"Small model params: {small_model.count_params()}")

large_model = DeepQNetwork(hidden1_size=256, hidden2_size=128)
print(f"Large model params: {large_model.count_params()}")
```

### Phase 3: Training (Days 3-4)

**Goal**: Train the agent to play Tetris

#### Reading
1. **STEP4_TRAINING.md**: Read "Key Concepts" section
2. **README.md**: Read "Detailed Step Guide - Step 4"

#### Hands-On: Quick Training
```bash
# Train for just 100 episodes (5-10 minutes)
python code/step4_train_dqn.py --num_epochs 100
```

**Observe**:
- [ ] Loss values printed every 10 episodes
- [ ] Score starting at ~50, improving gradually
- [ ] Model checkpoint saved at 100 episodes

**Questions to answer**:
- [ ] Why does loss decrease over time?
- [ ] How does epsilon decay?
- [ ] Why use a replay buffer?
- [ ] What's the target network for?

#### Hands-On: Full Training
```bash
# Train for 300 episodes (30-40 minutes)
# Run in background if possible
python code/step4_train_dqn.py --num_epochs 300
```

**Monitor progress**:
```bash
# Check saved models
ls -lh models/
```

**Code to understand**:
- Epsilon-greedy selection
- Replay buffer sampling
- Bellman equation implementation
- Target network update

### Phase 4: Evaluation (Days 4-5)

**Goal**: Test trained model and understand results

#### Reading
1. **STEP5_TESTING.md**: Read entire document

#### Hands-On: Test Your Model
```bash
# Test the trained model
python code/step5_test_model.py --model_path models/tetris_final.pth --num_games 20
```

**Analyze output**:
- [ ] Average score
- [ ] Lines cleared statistics
- [ ] Consistency (std dev)
- [ ] Performance rating

**Compare models**:
```bash
# Test intermediate checkpoints
python code/step5_test_model.py --model_path models/tetris_100.pth --num_games 20
python code/step5_test_model.py --model_path models/tetris_final.pth --num_games 20
```

**Visual inspection**:
```bash
# Watch the agent play
python code/step5_test_model.py --model_path models/tetris_final.pth --infinite --speed 0.5
```

### Phase 5: Experimentation (Days 5+)

**Goal**: Deep understanding through modification

#### Experiment 1: Different Network Sizes

Edit `step3_neural_network.py`:
```python
# Try smaller network
model = DeepQNetwork(hidden1_size=32, hidden2_size=16)

# Or larger network
model = DeepQNetwork(hidden1_size=256, hidden2_size=128)
```

Train and compare:
```bash
python code/step4_train_dqn.py --num_epochs 200
python code/step5_test_model.py --model_path models/tetris_final.pth --num_games 20
```

**Question**: How does network size affect learning?

#### Experiment 2: Different Learning Rates

```bash
# Very low learning rate
python code/step4_train_dqn.py --num_epochs 300 --lr 0.0001

# Higher learning rate  
python code/step4_train_dqn.py --num_epochs 300 --lr 0.005
```

**Question**: How does learning rate affect training curve?

#### Experiment 3: Different Batch Sizes

```bash
# Small batch
python code/step4_train_dqn.py --num_epochs 300 --batch_size 128

# Large batch
python code/step4_train_dqn.py --num_epochs 300 --batch_size 1024
```

**Question**: How does batch size affect stability?

#### Experiment 4: Reward Shaping

Edit `step1_tetris_basic.py`, modify the rewards:
```python
# Experiment with different penalty weights
reward -= 0.1 * (height / 20.0)      # Less height penalty
reward -= 2.0 * (holes / 50.0)       # More hole penalty
reward -= 0.3 * (bumpiness / 100.0)  # Less bumpiness penalty
```

**Question**: How does reward shaping affect agent behavior?

## 🎓 Key Concepts Checklist

### Tetris Game Concepts
- [ ] Board representation (20×10 grid)
- [ ] 7 tetromino pieces and rotations
- [ ] Collision detection
- [ ] Line clearing and scoring
- [ ] State features (lines, holes, bumpiness, height)
- [ ] Action representation (position, rotation)

### Reinforcement Learning Concepts
- [ ] Agent-environment interaction
- [ ] States, actions, rewards
- [ ] Episodic vs continuous learning
- [ ] Episode: sequence of steps until game over
- [ ] Trajectory: sequence of state-action-reward tuples

### Q-Learning Concepts
- [ ] Q-value: expected future reward from state-action pair
- [ ] Bellman equation: Q(s,a) = r + γ × max_Q(s',a')
- [ ] Q-table vs Q-network (when state space is large)
- [ ] Temporal difference (TD) learning
- [ ] Bootstrapping: using network's own predictions as targets

### Deep Q-Network Concepts
- [ ] Approximating Q-values with neural networks
- [ ] Experience replay: breaking correlation in data
- [ ] Target network: stable targets for training
- [ ] Epsilon-greedy: exploration vs exploitation tradeoff
- [ ] Reward shaping: encouraging desired behavior

### Neural Network Concepts
- [ ] Layers: input, hidden, output
- [ ] Weights and biases: learnable parameters
- [ ] Activation functions: ReLU adds non-linearity
- [ ] Forward pass: input → output
- [ ] Backward pass: computing gradients
- [ ] Gradient descent: updating weights to reduce loss
- [ ] Batch processing: efficient training on multiple samples

### Training Concepts
- [ ] Hyperparameters: learning rate, batch size, gamma, epsilon decay
- [ ] Loss function: how to measure prediction error
- [ ] Optimization: Adam, SGD, etc.
- [ ] Overfitting: model memorizes training data
- [ ] Convergence: model stops improving significantly

## 💡 Learning Tips

### 1. Read Code Comments

The code is well-commented. Reading comments + code is often better than reading documentation alone.

```bash
# Read through the main classes
less code/step1_tetris_basic.py  # Focus on TetrisGame class
less code/step3_neural_network.py  # Focus on DeepQNetwork class
less code/step4_train_dqn.py  # Focus on DQNAgent.train_step() method
```

### 2. Add Print Statements

Modify code to print intermediate values:

```python
# In step4_train_dqn.py, add after training step:
print(f"State shape: {states.shape}")
print(f"Q_pred range: [{q_pred.min():.2f}, {q_pred.max():.2f}]")
print(f"Q_target range: [{q_target.min():.2f}, {q_target.max():.2f}]")
print(f"Batch loss: {loss.item():.4f}")
```

### 3. Visualize Data

Create simple plots to understand patterns:

```python
import matplotlib.pyplot as plt

# Track loss over training
losses = []
# ... during training append loss ...
plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Step")
plt.show()
```

### 4. Ask "Why?" Questions

For each line of code:
- Why is this here?
- What happens if I remove it?
- What happens if I change it?

### 5. Compare to Baselines

Always compare your trained model to:
- **Random baseline**: agent takes random actions
- **Greedy baseline**: agent uses heuristics (e.g., minimize height)

```bash
# Random agent (Step 1)
python code/step1_tetris_basic.py  # Shows random play

# Your trained agent (Step 5)
python code/step5_test_model.py --model_path models/tetris_final.pth --num_games 10
```

## 🔍 Debugging Workflow

If something doesn't work:

1. **Check the error message**
   ```bash
   # Read the full error traceback
   # Look for line number and error type
   ```

2. **Verify file paths**
   ```bash
   ls -la models/tetris_final.pth  # Does it exist?
   ```

3. **Test with simpler code**
   ```python
   # Simplified test outside main training loop
   import torch
   from step3_neural_network import DeepQNetwork
   model = DeepQNetwork()
   print(f"Model created: {model}")
   ```

4. **Check dimensions**
   ```python
   # Print shapes of tensors at each step
   print(f"states shape: {states.shape}")
   print(f"q_pred shape: {q_pred.shape}")
   print(f"q_target shape: {q_target.shape}")
   ```

5. **Verify numerical stability**
   ```python
   # Check for NaN, Inf
   print(f"Any NaN in loss? {torch.isnan(loss).any()}")
   print(f"Loss is finite? {torch.isfinite(loss).all()}")
   ```

## 📊 Expected Learning Curve

After completing the full pipeline:

**Episode 0-50**: 
- Mostly random play
- Score ~50-100
- Many game overs

**Episode 50-200**:
- Agent starting to learn patterns
- Score ~200-500
- Less frequent game overs

**Episode 200-300**:
- Agent strategy emerging
- Score ~600-1200
- Reasonable play

**Episode 300+**:
- Model converged
- Score ~1000-2000
- Consistent strategy

## 🎯 Success Criteria

You've successfully learned this material when you can:

### Knowledge
- [ ] Explain what Q-learning is
- [ ] Explain how experience replay helps training
- [ ] Explain why we use target networks
- [ ] Explain epsilon-greedy exploration
- [ ] Explain the Bellman equation

### Implementation
- [ ] Understand all code in Step 1 (game engine)
- [ ] Understand all code in Step 3 (neural network)
- [ ] Understand main training loop in Step 4
- [ ] Train a model that plays better than random
- [ ] Test and interpret model statistics

### Experimentation
- [ ] Modify hyperparameters and observe effects
- [ ] Compare different network architectures
- [ ] Analyze why certain board states are easy/hard
- [ ] Suggest improvements to the algorithm

### Teaching
- [ ] Explain the project to someone else
- [ ] Answer questions about how parts work
- [ ] Debug issues in someone else's code

## 🚀 Next Steps After Mastery

Once you've mastered this project:

### Understand Variants
- **Dueling DQN**: Separate value and advantage streams
- **Double DQN**: Use separate network for target Q-value selection
- **Prioritized Experience Replay**: Sample important experiences more often
- **Rainbow DQN**: Combine all improvements

### Apply to Other Games
- **Atari games**: Similar concept with pixel input
- **Simple games**: CartPole, MountainCar (classic RL benchmarks)
- **Your own game**: Design and train agent for any game

### Improve the Tetris Agent
- **More features**: Add rotations, landing height, etc.
- **Different rewards**: Change scoring to encourage specific behavior
- **Online learning**: Real-time adaptation
- **Attention mechanism**: Learn which features matter

### Explore Other RL Algorithms
- **Policy Gradient**: Learn policy directly (A3C, PPO)
- **Actor-Critic**: Combine value and policy learning
- **Model-Based RL**: Learn environment model
- **Offline RL**: Learning from fixed dataset

## 📚 Additional Resources

### Reading
- Sutton & Barto: "Reinforcement Learning" (https://incompleteideas.net/book)
- Goodfellow, Bengio, Courville: "Deep Learning"
- Original DQN paper: "Human-level control through deep RL" (Nature 2015)

### Code Resources
- PyTorch documentation: https://pytorch.org/docs
- OpenAI Gym (other RL environments): https://gym.openai.com
- Stable Baselines3 (RL implementations): https://stable-baselines3.readthedocs.io

### Interactive Learning
- David Silver's RL course videos
- Spinning Up in Deep RL (OpenAI): https://spinningup.openai.com
- PyTorch tutorials: https://pytorch.org/tutorials

## 💬 Common Questions

**Q: Why does training sometimes diverge (loss becomes NaN)?**
A: Usually too high learning rate. Reduce lr flag and retry.

**Q: How long should training take?**
A: 300 episodes ~30-40 minutes on CPU, 5-10 minutes on GPU.

**Q: Can I train with a smaller network?**
A: Yes! Smaller networks train faster. Trade-off: might not learn complex patterns.

**Q: Why is my model worse than baseline?**
A: Could be: (1) not enough training, (2) bad hyperparameters, (3) code bug. Test intermediate models.

**Q: Can I save training progress and resume?**
A: Not built-in, but you can modify code to save optimizer state.

---

**Good luck on your learning journey! 🚀**

Remember: the goal is understanding, not perfect performance. Experiment, ask questions, and have fun!
