# Step 5: Testing & Evaluation

## Overview

Test your trained model on fresh Tetris games to evaluate performance. This step doesn't involve training—just pure evaluation.

## Running Step 5

### Basic Testing (10 Games)

```bash
python code/step5_test_model.py --model_path models/tetris_final.pth
```

Tests the model on 10 games and shows statistics

### Multiple Games

```bash
python code/step5_test_model.py --model_path models/tetris_final.pth --num_games 50
```

Play 50 games for more reliable statistics

### Interactive Mode

```bash
python code/step5_test_model.py --model_path models/tetris_final.pth --infinite
```

Infinite game loop with controls:
- **SPACE**: Pause/resume
- **R**: Restart current game
- **Z**: Rewind last move
- **Q**: Quit

### Adjust Speed

```bash
python code/step5_test_model.py --model_path models/tetris_final.pth --infinite --speed 0.5
```

Slower visualization (default speed is 2.0x)

## Understanding the Output

### Game-by-Game Results

```
Game  1: Score   1250 | Pieces  45 | Lines  25 | Steps  180
Game  2: Score    980 | Pieces  38 | Lines  20 | Steps  150
Game  3: Score   1540 | Pieces  51 | Lines  32 | Steps  210
...
```

**Columns**:
- **Score**: Total points earned (clearing lines gives points)
- **Pieces**: Number of pieces placed before game over
- **Lines**: Number of complete lines cleared
- **Steps**: Number of moves made

**Interpretation**:
- Higher score = better
- More pieces = longer games = better strategy
- More lines = efficiently clearing rows = better strategy
- More steps = last longer before game over = better

### Statistics Summary

```
════════════════════════════════════════════════════════
📊 STATISTICS
════════════════════════════════════════════════════════

Score   : μ=1245.3 σ=180.5 | median=1280.0 | min=650 max=1890
Pieces  : μ=45.2   σ=6.3  | median=45.0  | min=32  max=58
Lines   : μ=24.8   σ=4.1  | median=25.0  | min=15  max=35
Steps   : μ=182.4  σ=28.7 | median=185.0 | min=120 max=245

────────────────────────────────────────────────────────
🌟 Excellent (avg lines > 100)!
────────────────────────────────────────────────────────
```

**Statistics explained**:

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| **μ** (mean) | Average across all games | Higher is better |
| **σ** (std dev) | Consistency (lower = more consistent) | Lower is better |
| **median** | Middle value | Similar to mean is good |
| **min** | Worst game | Low variance is good |
| **max** | Best game | Shows model's potential |

**Performance rating**:
- **Excellent**: Average lines > 100
- **Good**: Average lines > 50
- **Can improve**: Average lines ≤ 50

### Interpreting Results

#### High average, low std dev
```
Score: μ=1500 σ=50
```
✅ **Excellent**: Agent plays consistently well

#### High average, high std dev
```
Score: μ=1200 σ=300
```
⚠️ **Inconsistent**: Sometimes great, sometimes bad (lucky/unlucky board)

#### Low average
```
Score: μ=300 σ=100
```
❌ **Needs more training**: Agent hasn't learned well enough

## Model Selection

### Which Model to Test?

Models are saved at intervals during training:

```
models/
├── tetris_100.pth      # After 100 episodes
├── tetris_200.pth      # After 200 episodes
├── tetris_300.pth      # After 300 episodes
└── tetris_final.pth    # After all episodes
```

**Strategy**:
1. Test `tetris_100.pth` - see early learning
2. Test `tetris_300.pth` - see mid-training
3. Test `tetris_final.pth` - see best model

**Observation**:
- Earlier models: Lower scores, more variation
- Later models: Higher scores, more consistent

### Finding the Best Model

```bash
# Test all models and compare
python code/step5_test_model.py --model_path models/tetris_100.pth --num_games 20
python code/step5_test_model.py --model_path models/tetris_300.pth --num_games 20
python code/step5_test_model.py --model_path models/tetris_final.pth --num_games 20
```

Compare statistics to see which model performs best.

## Analysis Workflow

### 1. Quick Test

```bash
python code/step5_test_model.py --model_path models/tetris_final.pth --num_games 10
```

Fast test to see if model works at all

### 2. Full Evaluation

```bash
python code/step5_test_model.py --model_path models/tetris_final.pth --num_games 50
```

Larger sample for more reliable statistics

### 3. Visual Inspection

```bash
python code/step5_test_model.py --model_path models/tetris_final.pth --infinite --speed 0.5
```

Watch the agent play, see if strategy makes sense

### 4. Compare Training Models

Test multiple checkpoints from training to see learning trajectory

## Expected Results

### Untrained Model (Random Network)

```
Score   : μ=100   σ=50  | median=90    | min=20  max=200
Pieces  : μ=6     σ=3   | median=6     | min=2   max=12
Lines   : μ=3     σ=1   | median=3     | min=1   max=6
```

Very poor because network outputs random Q-values

### Partially Trained (100 episodes)

```
Score   : μ=250   σ=100 | median=220   | min=50  max=500
Pieces  : μ=12    σ=5   | median=11    | min=3   max=25
Lines   : μ=6     σ=2   | median=6     | min=2   max=12
```

Some learning but still struggling

### Well Trained (300+ episodes)

```
Score   : μ=1200  σ=200 | median=1250  | min=600 max=1900
Pieces  : μ=45    σ=8   | median=46    | min=25  max=65
Lines   : μ=24    σ=4   | median=24    | min=12  max=35
```

Good strategy, consistent play

### Excellent Model (500+ episodes)

```
Score   : μ=1800  σ=150 | median=1850  | min=1200 max=2400
Pieces  : μ=70    σ=7   | median=71    | min=50  max=90
Lines   : μ=36    σ=3   | median=36    | min=28  max=45
```

Excellent strategy, very consistent

## Debugging Poor Performance

### Problem: Very Low Scores

**Possible causes**:
1. Model didn't train properly
2. Testing wrong model file
3. Model file is corrupted

**Fix**:
```bash
# Verify file exists and is loadable
python -c "import torch; torch.load('models/tetris_final.pth')"

# Check model was saved with training
# (Size should be ~30KB for DQN)
ls -lh models/tetris_final.pth

# Re-train with more epochs
python code/step4_train_dqn.py --num_epochs 500
```

### Problem: High Variance (Inconsistent Results)

**Possible cause**: Model overfits to specific board states

**Observation**: Some games very good (~1500 score), some very bad (~200 score)

**Fix**:
```bash
# Use smaller network (less overfitting)
# Edit step3_neural_network.py:
# model = DeepQNetwork(hidden1_size=32, hidden2_size=16)

# Or train longer to generalize better
python code/step4_train_dqn.py --num_epochs 1000
```

### Problem: Model Plays Same Moves

**Observation**: Model places pieces in same position repeatedly (bad strategy)

**Cause**: Network didn't learn to evaluate different actions

**Fix**:
```bash
# Increase exploration during training
python code/step4_train_dqn.py --num_epochs 500 --decay_epochs 3000

# Or try different learning rate
python code/step4_train_dqn.py --num_epochs 500 --lr 0.0005
```

## Visual Analysis

When you run with `--infinite`, watch the model play:

### Good Model Behavior
- ✅ Pieces placed to minimize holes
- ✅ Attempts to keep stack low
- ✅ Fills gaps strategically
- ✅ Clears multiple lines at once

### Poor Model Behavior
- ❌ Places pieces randomly
- ❌ Creates many holes
- ❌ Stack grows very tall
- ❌ Doesn't clear lines efficiently
- ❌ Game overs quickly

## Comparing Models

### Create Comparison Table

```bash
# Test multiple models
for model in models/tetris_{100,200,300,400,500}.pth; do
    echo "Testing $model..."
    python code/step5_test_model.py --model_path "$model" --num_games 20
done
```

Creates table of how performance improves with training

## Advanced Analysis

### Analyzing Q-Values

During testing, the model predicts Q-values for each action:

```python
# In step5_test_model.py, modify to print Q-values
action, q_value = self.select_best_action(env)
print(f"Best action: {action}, Q-value: {q_value:.2f}")

# Higher Q-value = more confident prediction
# Low Q-values suggest uncertain decisions
```

### Testing on Different Board States

Modify the game to start with specific configurations:

```python
# In step1_tetris_basic.py
def set_board_state(self, board):
    """Load a specific board configuration"""
    self.board = [row[:] for row in board]
    self._spawn_new_piece()

# Test on same board multiple times
game = TetrisGame()
game.set_board_state(difficult_board)
# Run tests...
```

## Statistics Deep Dive

### Coefficient of Variation (CV)

```
CV = σ / μ
```

Measures consistency (lower is more consistent)

```python
import numpy as np
scores = [1250, 980, 1540, 1100, 1350]
cv = np.std(scores) / np.mean(scores)
print(f"CV: {cv:.2f}")  # Low CV = consistent
```

### Confidence Intervals

```python
import scipy.stats as stats
mean = np.mean(scores)
se = stats.sem(scores)  # Standard error
ci = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=se)
print(f"95% CI: [{ci[0]:.0f}, {ci[1]:.0f}]")
```

Shows likely true mean within range

## Reporting Results

When sharing results, include:

1. **Configuration used**:
   ```
   Trained for 300 episodes
   Learning rate: 0.001
   Batch size: 512
   ```

2. **Test parameters**:
   ```
   Tested on 50 games
   Model: tetris_final.pth
   ```

3. **Results**:
   ```
   Average score: 1245 ± 180
   Average lines: 24.8 ± 4.1
   Performance: Good (avg lines > 50)
   ```

4. **Observations**:
   ```
   Model consistently clears lines but struggles with tall stacks.
   Sometimes creates holes that lead to early game over.
   ```

## Key Takeaways

1. **Use multiple games for statistics**: Single games are noisy, test 20+ games

2. **Track multiple metrics**: Score, pieces, lines all tell different stories

3. **Compare checkpoints**: See learning progress by testing models from different training stages

4. **Visual inspection matters**: Watch the model play to understand strategy

5. **Interpret variance**: High std dev means inconsistent, low std dev means reliable

6. **Reasonable expectations**: Even good models won't be perfect (humans struggle too!)

## Next Steps After Testing

### If Performance is Good
- Try more advanced architectures (dueling DQN, double DQN)
- Experiment with different reward functions
- Test on different board sizes

### If Performance is Poor
- Check training logs—did loss decrease?
- Test intermediate models from training
- Try different hyperparameters
- Increase training episodes

### For Learning
- Analyze which board states the model struggles with
- Visualize Q-value predictions
- Compare to human play
- Try different state representations
