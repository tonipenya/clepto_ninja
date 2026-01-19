# 🥷 CleptoNinja

CleptoNinja is a small experimental project built around a **custom card game** I’m designing.

The goal is twofold:

- **Stress-test and refine the game mechanics** by playing it far more than any human group reasonably could.
- **Learn, hands-on, how modern multi-agent learning techniques actually behave** once you leave toy examples.

This repo is intentionally practical, a bit messy, and focused on learning by doing.

## What’s in here

- A deterministic, multi-player card game engine.
- Several baseline players (random, greedy).
- A self-play training setup using **PPO** and population-based opponents.
- Evaluation code focused on _winning matches_, not just scoring points.
- Enough tooling to iterate on rules and strategies quickly.

No claims of optimal play. No magic abstractions. Just experiments.

## Run the game

### Locally

```sh
pip install -r requirements.txt
python play.py
```

### Via Docker

```sh
make run
```

## Train AI players

Open and run:

```
make train
```

This will train agents via self-play and save a model checkpoint, e.g.:

[best_actor_critic_player.pt](./checkpoints/best_actor_critic_player.pt)

The checkpoints [directory](./checkpoints/) will also contain the checkpoints
for the best 20 models.

Training is intentionally transparent and easy to tweak.

## Build Docker image

```sh
make build
```

## Why this exists

I wanted a project where:
• the **rules are mine**,
• the learning signal is messy and imperfect,
• and improvements come from understanding the system, not tuning a benchmark.

If you’re into game design or RL, this kind of project is a lot of fun.

⸻

This is a hobby project. Expect experiments, not polish.
