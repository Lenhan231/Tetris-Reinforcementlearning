# Step 1: Tetris Game Engine

## Overview

Build a complete Tetris game with proper physics, collision detection, and state representation. This step creates the foundation that the AI agent will play.

## Game Basics

### Board
- **Dimensions**: 20 rows × 10 columns
- **Coordinates**: (0,0) is top-left, y increases downward, x increases rightward

### Pieces (Tetrominos)

7 types of falling blocks, each with 4 rotation states:

```
O-Piece (Yellow)    T-Piece (Purple)    S-Piece (Green)     Z-Piece (Red)
██                  ███                 ·██                 ██·
██                  ·█·                 ██·                 ·██

I-Piece (Cyan)      L-Piece (Orange)    J-Piece (Blue)
████                ···█                █··
                    ███                 ███
```

### Game Mechanics

1. **Spawn**: New piece appears at top-center of board
2. **Fall**: Piece drops one row per action
3. **Collision**: Detect when piece hits bottom or another block
4. **Lock**: Piece becomes part of board when it can't move down
5. **Clear Lines**: Remove completely filled rows
6. **Score**: Award points for cleared lines
7. **Game Over**: When a new piece can't fit at spawn position

## Running Step 1

```bash
python code/step1_tetris_basic.py
```

This demonstrates the game engine with 5 random moves. You'll see:
- Board visualization with blocks
- Current score and statistics
- State features extracted from the board

## Key Classes and Methods

### TetrisGame

Main game class that manages all Tetris logic.

#### Initialization
```python
game = TetrisGame(height=20, width=10)
```

#### Main Methods

**reset()**
- Reset game to initial state
- Returns: state features tuple (lines, holes, bumpiness, height)

**step(action)**
- Execute one action (place and drop a piece)
- Input: `action = (x_pos, num_rotations)`
  - `x_pos`: Column where piece lands (0-10)
  - `num_rotations`: How many times to rotate (0-3)
- Returns: `(reward, game_over, state_features)`

**get_next_states()**
- Pre-compute all possible next states from current piece
- Returns: Dictionary mapping actions to resulting state features
- Used by AI agent to plan moves

**print_board()**
- Display board in console (█ = block, · = empty)
- Shows current score and state metrics

**render()**
- Visualize board using OpenCV
- Shows colors, grid, and info panel

## State Features

The AI agent sees 4 features extracted from the board:

### 1. Lines Cleared
- **Definition**: Total number of rows cleared in this game
- **Good value**: High (more lines = more reward)
- **Range**: 0 to 20

### 2. Holes
- **Definition**: Empty spaces under filled cells in a column
- **Calculation**: For each column, count empty cells below the highest block
- **Good value**: Low (holes block future placements)
- **Range**: 0 to 200

### 3. Bumpiness
- **Definition**: Height variation between adjacent columns
- **Calculation**: Sum of absolute differences between column heights
- **Good value**: Low (smooth surface easier to manage)
- **Range**: 0 to 200

### 4. Height
- **Definition**: Total height of the stack (sum of all column heights)
- **Calculation**: Sum of heights across all 10 columns
- **Good value**: Low (tall stacks lead to game over)
- **Range**: 0 to 200

### How to Extract Features

```python
features = game._get_state_features()
lines, holes, bumpiness, height = features

# Interpret:
# - lines: higher is better
# - holes, bumpiness, height: lower is better
```

## Internal Mechanics

### Collision Detection

Before placing a piece, check if any block overlaps with:
1. Board boundaries (left, right, bottom)
2. Already placed blocks

```python
if game._check_collision(piece, x, y):
    # Collision detected
    pass
```

### Rotation

Pieces rotate 90° clockwise using matrix transpose + reverse:

```
[1, 0]      [1, 0]
[1, 1]  →   [0, 1]  (after transpose and reverse rows)
```

### Line Clearing

1. Identify all full rows (no empty cells)
2. Delete full rows
3. Insert empty rows at top (gravity effect)
4. Calculate score: `1 + (num_lines²) × width`

### Reward Shaping

The reward given for each move encourages good board states:

```python
reward = points_from_cleared_lines
reward -= 0.5 * (height / 20.0)      # Penalize tall stacks
reward -= 1.0 * (holes / 50.0)       # Penalize holes
reward -= 0.7 * (bumpiness / 100.0)  # Penalize bumpy surface
if game_over:
    reward -= 10  # Strong penalty for game over
```

## Usage Examples

### Basic Game Loop

```python
game = TetrisGame()
state = game.reset()  # (lines, holes, bumpiness, height)

while not game.game_over:
    # Get all possible moves
    possible_moves = game.get_next_states()
    
    # Choose an action (e.g., random)
    action = random.choice(list(possible_moves.keys()))
    
    # Execute action
    reward, done, state = game.step(action)
    
    # Visualize
    game.print_board()
```

### Planning Ahead

```python
# Before making a move, see all possible next states
next_states = game.get_next_states()
# next_states = {
#   (0, 0): (lines=5, holes=2, bump=10, height=15),
#   (1, 0): (lines=5, holes=1, bump=8, height=14),
#   (2, 1): (lines=6, holes=0, bump=5, height=12),
#   ...
# }

# Choose best action based on features
best_action = max(next_states.keys(), 
                  key=lambda a: next_states[a][0] - 0.5*next_states[a][3])
reward, done, state = game.step(best_action)
```

## Common Patterns

### Play N Random Games

```python
scores = []
for game_num in range(10):
    game = TetrisGame()
    game.reset()
    
    while not game.game_over:
        actions = list(game.get_next_states().keys())
        action = random.choice(actions)
        game.step(action)
    
    scores.append(game.score)

print(f"Average score: {sum(scores)/len(scores)}")
```

### Visualize a Game

```python
game = TetrisGame()
game.reset()

while not game.game_over:
    next_states = game.get_next_states()
    action = random.choice(list(next_states.keys()))
    reward, done, state = game.step(action)
    game.render(window_name="Tetris Game")
    time.sleep(0.1)  # Slow down to watch
```

## Key Takeaways

1. **State representation matters**: The 4 features (lines, holes, bumpiness, height) encode important aspects of board quality

2. **Actions are pre-computed**: Use `get_next_states()` to see all possibilities before deciding

3. **Reward includes penalties**: Clearing lines is good, but the agent also learns to avoid bad board states

4. **Collision physics are key**: Proper collision detection is essential for realistic gameplay

5. **Board size is small**: 20×10 grid is manageable for an AI to learn from

## Next Steps

Once you understand the game engine:
- Move to **Step 3** to learn about neural networks
- Then **Step 4** to train the agent
- Finally **Step 5** to evaluate the trained model
