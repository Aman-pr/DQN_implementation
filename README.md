# Flappy Bird (Simplified) — DQN Agent

A minimal reinforcement learning project where a Deep Q-Network (DQN) learns to play a **simplified Flappy Bird** environment. The env is lightweight (no graphics) and uses shaped rewards so the agent can learn to survive and pass pipes.

<p align="center">
  <em>Train on CPU in minutes; extend to Double/Dueling/Per later.</em>
</p>

---

## ✨ Features

* Tiny, self-contained environment (no Gym required)
* DQN with target network & replay buffer
* Reward shaping for faster learning (survival + gap-centering + pipe pass, death penalty)
* Easy hyperparameters you can tune from CLI
* Save/Load model for evaluation

---

## 📦 Project Structure

```
flappy-bird-dqn/
├── flappy_dqn.py         
├── requirements.txt      
├── .gitignore            
└── README.md              
```

> If you only want the bare minimum, keep just `flappy_dqn.py` and `requirements.txt`.

---

## 🧠 Algorithm

This project implements a vanilla **DQN**:

* Q-network: 2 hidden layers (default 64→64; recommended 128→128)
* Target network for stable bootstrapping
* Replay buffer with uniform sampling
* ε-greedy exploration with slow decay

**Reward shaping** (recommended default):

* `+0.1` per step survived
* `+5.0` when a pipe is passed
* `-5.0` on death
* *(Optional, stronger shaping)*: small penalty for distance from the gap center each step

---

## 🚀 Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

<details>
<summary>requirements.txt</summary>

```
numpy
torch>=2.0.0
pygame>=2.5.0
```

</details>

### 2) Train
```bash
python flappy_dqn.py --episodes 2000
```

During training, you’ll see logs like:

```
Ep 123, Score 2, Reward 7.40, Avg20 1.25, Eps 0.742
```

* **Score**: pipes passed in that episode
* **Reward**: sum of shaped rewards
* **Avg20**: moving average of last 20 episode scores
* **Eps**: current ε for ε-greedy

### 3) Evaluate

```bash
python flappy_dqn.py --eval --model flappy_dqn.pt
```

* Loads the saved model and runs greedy play (ε=0)

---

## ⚙️ CLI Options

```
--episodes INT        # number of training episodes (default 2000)
--eval                # run evaluation instead of training
--model PATH          # model path for save/load (default flappy_dqn.pt)
```

---

## 🔧 Tuning Tips (to beat >3 pipes)

1. **Network size**: bump to `128→128` (or `256→256`) in `DQN`.
2. **Exploration**: slow decay `eps_decay=0.999`, set floor `eps_min=0.1`.
3. **Extra shaping**: penalize distance to gap center each step:

   ```python
   # inside env.step()
   reward -= abs(self.bird_y - self.pipe_gap_y) * 0.05
   ```
4. **Replay buffer**: increase to `200_000` to diversify experiences.
5. **More training**: try 5k–10k episodes; Flappy Bird is sparse & timing-heavy.
6. **Advanced**: implement **Double DQN**, **Dueling**, **PER** for bigger gains.

---

## 🧩 Minimal Code Snippet (Env + Reward Shaping)

> Full code lives in `flappy_dqn.py`; this snippet shows the key reward changes.

```python
# reward shaping inside step()
reward = 0.1  # survival
passed_pipe = False

if self.pipe_x < 0:
    self.pipe_x = 1.0
    self.pipe_gap_y = np.random.uniform(0.3, 0.7)
    passed_pipe = True
    reward += 5.0  # pipe passed

# (optional) encourage staying near gap center
reward -= abs(self.bird_y - self.pipe_gap_y) * 0.05

if collision:
    reward -= 5.0  # death penalty
```

---

## 🧪 Expected Results

* With the defaults, expect average score around **1–3 pipes** within \~2k episodes.
* With tuning above (bigger net + slower ε decay + distance penalty), average can push further.

> For reproducible comparisons, fix a random seed and report Avg20.

---

## 🛠 Troubleshooting

* **`ModuleNotFoundError: numpy/torch`** → run `pip install -r requirements.txt` inside the same environment you call `python` from. Verify with `which python` and `which pip`.
* **Very low scores** → increase episodes, slow ε decay, use 128→128 hidden units, add distance-to-gap penalty.
* **Torch install slow** → CPU wheels install quickly; for CUDA use PyTorch’s extra index URL matching your driver.

---

## 📝 .gitignore

```
__pycache__/
*.pyc
*.pt
.venv/
.env/
.DS_Store
```

---

## 📄 License (MIT)

```
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 Contributing

* Issues and PRs welcome! Please include: env details, command used, and a short log snippet.

---

## 📣 Citation

If this repo helps your work or teaching, consider citing it:

```
@misc{flappy_dqn_minimal,
  title  = {Flappy Bird (Simplified) — DQN Agent},
  author = {Your Name},
  year   = {2025},
  note   = {Minimal no-graphics environment with reward shaping}
}
```

---

### ⭐ If you publish this on GitHub

1. Create a new repo and add these files.
2. Paste this README as `README.md`.
3. Commit `flappy_dqn.py` and `requirements.txt`.
4. (Optional) Add `LICENSE` and `.gitignore` from above.
5. Push and share! 🚀
