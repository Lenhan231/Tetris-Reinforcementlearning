# 🎮 Tetris Deep Q-Learning

A Deep Q-Network (DQN) agent trained to play Tetris via reinforcement learning.

## 📋 Project Structure

```
tetris_from_scratch/
├── code/
│   ├── tetris.py          # Environment + feature extraction
│   ├── network.py         # DQN architecture
│   ├── train.py           # Training pipeline
│   ├── wandb_config.py    # WandB setup
│   └── test.py             # Evaluation
├── models/                 # Saved checkpoints
└── README.md
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 🏋️ Training

```bash
python code/train.py                # basic
python code/train.py --render       # with rendering
python code/train.py --wandb        # with WandB tracking
```

All parameters are configurable via CLI args. Best known config:

```bash
python code/train.py --wandb \
  --shape_holes -1 --shape_bump -1 --shape_height -1 \
  --num_epochs 3000 --memory_size 3000 \
  --max_episode_pieces 100000 --lr 0.001 \
  --target_update 10 --decay_epochs 2000 --final_eps 0.001
```

## 🧪 Testing

```bash
python code/test.py --model_path models/tetris_best_{score}.pth
```

- Default: reports statistics over 10 games
- `--infinite`: renders gameplay continuously

```bash
python code/test.py --model_path models/tetris_best_100000.pth --infinite
```

## 🎛️ Hyperparameters

| Parameter | Default | Description |
|---|---:|---|
| num_epochs | 3000 | Training episodes |
| batch_size | 512 | Training batch size |
| lr | 0.001 | Learning rate |
| gamma | 0.99 | Discount factor |
| initial_eps | 1.0 | Initial exploration |
| final_eps | 0.01 | Final exploration |
| memory_size | 3000 | Replay buffer size |
| max_episode_pieces | 2000 | Episode length limit |

**Reward shaping**

| Parameter | Default | Description |
|---|---:|---|
| shape_holes | -1.0 | Hole penalty |
| shape_bump | -1.0 | Bumpiness penalty |
| shape_height | -1.0 | Height penalty |

## 📊 State Features

| Feature | Description |
|---|---|
| lines_cleared | Number of cleared lines |
| holes | Empty spaces below blocks |
| bumpiness | Height difference between columns |
| height | Total stack height |

## 🧠 Model Architecture

See `code/network.py`.

## ⚙️ Training Method

- **Experience Replay** — random batches from stored transitions
- **Target Network** — periodically updated for stability
- **Epsilon-Greedy** — exploration → exploitation
- **Reward Shaping** — guides toward better board states

## 📦 Outputs

```
models/tetris_best_XXXXX.pth   # checkpoints
code/wandb/                    # WandB logs
```

## 🛠️ Troubleshooting

**Training is slow**
```bash
python code/train.py --batch_size 256
```
Enable CUDA if available.

**Model stops improving**
```bash
python code/train.py --num_epochs 5000
```
Adjust reward shaping parameters.