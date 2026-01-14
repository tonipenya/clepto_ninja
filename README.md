# 🥷 CleptoNinja

## Run the game

### Locally

Make sure dependencies are installed:

```sh
pip install -r requirements.txt
python play.txt
```

### Via Docker (see build for instructions)

```sh
make run
```

## Train AI players

Run [train.ipynb](./train.ipynb). It'll generate a model [checkpoint](./best_actor_critic_player.pt).

## Build Docker image

```sh
make build
```
