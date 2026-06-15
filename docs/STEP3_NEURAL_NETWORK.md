# Step 3: Deep Q-Network Architecture

## Overview

Build a neural network that learns to estimate Q-values (expected future rewards) for Tetris game states. This is the "brain" of the AI agent.

## Running Step 3

```bash
python code/step3_neural_network.py
```

This demonstrates:
- Network architecture visualization
- Parameter counting
- Single state prediction example
- Batch processing
- Training step simulation with Bellman updates

## What is a Q-Network?

### Q-Value Definition

**Q(state, action)** = Expected cumulative future reward if we take this action in this state

Example:
- Board state: lots of holes, tall stack, 5 cleared lines
- Action: place piece at position 5, rotated twice
- Q-value: -50 (bad state leads to bad future)

vs.

- Board state: few holes, short stack, 10 cleared lines
- Action: place piece at position 3, no rotation
- Q-value: +150 (good state leads to good future)

### Why Use a Network?

**Without a network** (traditional Q-learning):
- Store Q-values in a lookup table: `Q[state][action]`
- Problem: Tetris state space is infinite! Can't create a table

**With a neural network**:
- Network learns pattern: state → estimated Q-value
- Works for any state, even ones never seen before
- Can generalize from limited experience

## Network Architecture

### Visual Representation

```
INPUT LAYER (4 features)
  [lines_cleared]
  [holes        ]
  [bumpiness    ]
  [height       ]
         ↓
  4 neurons

HIDDEN LAYER 1 (64 neurons)
  [Dense(4 → 64)]
  [ReLU activation]
         ↓
  64 neurons

HIDDEN LAYER 2 (32 neurons)
  [Dense(64 → 32)]
  [ReLU activation]
         ↓
  32 neurons

OUTPUT LAYER (1 Q-value)
  [Dense(32 → 1)]
  [Linear activation]
         ↓
  Q-value (scalar)
```

### Layer Breakdown

#### Input Layer
- **Size**: 4 features
- **Content**: (lines_cleared, holes, bumpiness, height)
- **Normalization**: Implicit (network learns to handle raw values)

#### Hidden Layer 1
- **Size**: 64 neurons
- **Activation**: ReLU (Rectified Linear Unit)
- **Purpose**: Learn low-level patterns about board states
- **Parameters**: 4×64 weights + 64 biases = 320 params

#### Hidden Layer 2
- **Size**: 32 neurons
- **Activation**: ReLU
- **Purpose**: Compress information, learn higher-level patterns
- **Parameters**: 64×32 weights + 32 biases = 2,080 params

#### Output Layer
- **Size**: 1 value (the Q-value)
- **Activation**: Linear (no activation, can output any number)
- **Purpose**: Estimate Q-value for the input state
- **Parameters**: 32×1 weights + 1 bias = 33 params

### Total Parameters

```
FC1: 320
FC2: 2,080
FC3: 33
────────
Total: ~2,500 parameters
```

Small network = fast training, less overfitting

## Activation Functions

### ReLU (Rectified Linear Unit)

```
f(x) = max(0, x)
```

**Graph**:
```
      │      ╱
      │     ╱
  f(x)│    ╱
      │   ╱
      │  ╱
      └──────────x
      │
```

**Benefits**:
- **Non-linearity**: Allows network to learn curved patterns
- **Efficiency**: Simple computation (just max operation)
- **Sparsity**: Many outputs are 0 (efficient)

**Without ReLU** (linear activation):
- Network becomes just linear combinations
- Can only learn linear patterns
- Not expressive enough for complex games

**Linear Output Layer**:
- No activation on final layer
- Q-values can be negative or positive
- Allows full range of value estimates

## How the Network Works

### Forward Pass (Prediction)

Input state → network → predicted Q-value

```python
state = torch.FloatTensor([10, 5, 20, 15])
# lines_cleared=10, holes=5, bumpiness=20, height=15

q_value = model(state)  # Returns something like 85.3
```

### Training (Learning)

Adjust weights so network outputs correct Q-values

**Bellman Equation** (target):
```
Q_target = reward + γ × max_Q(next_state)
```

**Training goal**: Make Q_predicted = Q_target

**Loss function**: Mean Squared Error (MSE)
```
Loss = (Q_target - Q_predicted)²
```

**Gradient descent**:
1. Compute loss
2. Backpropagate gradients through network
3. Update weights: `weight -= learning_rate × gradient`

### Example Training Step

```python
# Before training
state = [10, 5, 20, 15]
Q_predicted = 45.2
Q_target = 85.0  (from Bellman equation)
Loss = (85.0 - 45.2)² = 1,584

# After one update
Q_predicted = 48.5
Q_target = 85.0
Loss = (85.0 - 48.5)² = 1,332  # Loss decreased!

# After many updates
Q_predicted ≈ 85.0
Loss ≈ 0  # Network learned!
```

## Using the Network

### Single State Prediction

```python
from step3_neural_network import DeepQNetwork
import torch

# Create network
model = DeepQNetwork()
model.eval()  # Set to evaluation mode

# Predict Q-value for one state
state = torch.FloatTensor([10, 5, 20, 15])
with torch.no_grad():
    q_value = model(state)
    print(f"Q-value: {q_value.item():.4f}")
```

### Batch Prediction (Multiple States)

```python
# Multiple states at once (more efficient)
states = torch.FloatTensor([
    [10, 5, 20, 15],   # State A
    [8, 3, 18, 14],    # State B
    [12, 7, 22, 16],   # State C
])

with torch.no_grad():
    q_values = model(states)  # Shape: (3, 1)
```

### Action Selection

```python
# We have multiple possible next states from get_next_states()
next_states = {
    (0, 0): (5, 2, 10, 15),
    (1, 1): (5, 1, 8, 14),
    (2, 2): (6, 0, 5, 12),
}

# Predict Q-values for all
actions = list(next_states.keys())
features = [list(next_states[a]) for a in actions]
states_tensor = torch.FloatTensor(features)

with torch.no_grad():
    q_values = model(states_tensor).squeeze()

# Choose action with highest Q-value
best_idx = torch.argmax(q_values).item()
best_action = actions[best_idx]
```

## Customizing the Network

### Different Layer Sizes

```python
# Smaller network (for fast testing)
small_model = DeepQNetwork(hidden1_size=32, hidden2_size=16)

# Larger network (for complex learning)
large_model = DeepQNetwork(hidden1_size=256, hidden2_size=128)

# Custom input/output sizes
custom_model = DeepQNetwork(
    input_size=8,      # More features
    hidden1_size=128,
    hidden2_size=64,
    output_size=1      # Keep output as 1 (one Q-value per state)
)
```

### Counting Parameters

```python
model = DeepQNetwork()
print(f"Total parameters: {model.count_params():,}")

# Output: Total parameters: 2,533
```

## Weight Initialization

The network uses **Xavier initialization**:

```python
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
```

**Why?** Proper initialization helps training converge faster and more stably.

## Training vs Evaluation Mode

```python
# During training
model.train()
q_values = model(states)  # Dropout applied (if added)
loss = criterion(q_values, targets)
loss.backward()
optimizer.step()

# During evaluation/testing
model.eval()
with torch.no_grad():
    q_values = model(states)  # Faster, no dropout
    predictions = q_values
```

## Common Issues and Solutions

### Issue: Network Always Outputs Same Value

**Cause**: Not training or poor learning rate

**Solution**:
```python
# Check learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Default: good

# Check loss is decreasing
print(f"Loss: {loss.item()}")  # Should decrease over time
```

### Issue: Q-values are NaN or Inf

**Cause**: Learning rate too high, exploding gradients

**Solution**:
```python
# Use lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Or clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Issue: Network Overfits

**Cause**: Too many parameters, not enough data

**Solution**:
```python
# Use smaller network
model = DeepQNetwork(hidden1_size=32, hidden2_size=16)

# Or use larger replay buffer
agent.memory = deque(maxlen=50000)  # Store more experiences
```

## Key Takeaways

1. **Q-network maps states to expected rewards**: Enables generalization to unseen states

2. **Small network is good enough**: 2,500 parameters can learn Tetris strategy

3. **ReLU adds non-linearity**: Without it, network is just linear combinations

4. **Training uses Bellman equation**: Q_target = reward + γ × max_Q(next_state)

5. **Initialization matters**: Proper weight initialization speeds up training

6. **Batch processing is efficient**: Predict Q-values for multiple states at once

## Next Steps

- **Step 4**: Use this network in a training loop
- Experiment with architecture: try different layer sizes
- Try other optimizers: SGD, RMSprop instead of Adam
