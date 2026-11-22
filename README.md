# Sokoban solver

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/fahroz1/sokoban-solver-01-run-20261121/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/fahroz1/sokoban-solver-01-run-20261121/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver for the [make.fun bounty](https://make.fun/bounty/sokoban-solver-01-run-20261121) — one Node 22 program (`main.ts`) that solves 5 drawn Sokoban levels in a single run.

## How it works

Each level is decomposed into its rooms: the hallway row where the player starts is treated as a wall for box movement, and the connected components of the remaining floor are the rooms (3–4 boxes each). Every room is solved push-optimally with A* over pushes — canonical player regions, a minimum-assignment box-to-goal heuristic, and deadlock pruning (freeze, 2×2 blocks, wall segments with no goal). The rooms are then stitched together with shortest-path walks through the hallway. Total pushes equal the sum of the room optima, which is the level's push-optimal count by construction.

## Practice score

**1,000,000 / 1,000,000** on the kit's practice set — every instance solved in exactly its push-optimal count (i001: 13, i002: 16, i003: 29, i004: 47, i005: 60 pushes). The house baseline (plain-astar 1.0.0) scores 200,000 on the same set. One full practice run takes ~0.4 s (limit 60 s) and is fully deterministic.

## Run

```sh
MAKEFUN_INPUT_DIR=<input dir> MAKEFUN_OUTPUT_DIR=<output dir> node main.ts
```

Reads `<input>/<id>/level.txt` (XSB) for `i001`–`i005` and writes `<output>/<id>/solution.txt` (moves `u`, `d`, `l`, `r`).
