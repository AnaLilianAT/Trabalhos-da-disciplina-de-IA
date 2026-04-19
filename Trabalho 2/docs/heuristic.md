# Heuristic Module Guide

This document explains the reusable heuristics available in the project and how
to select them by name.

## Goals

The heuristic module serves two different use cases:

- State evaluation for Minimax-like search
- Move prioritization for MCTS rollouts

Even though both use similar signals, they optimize different decisions:

- State evaluation answers: "How good is this board position?"
- Rollout policy answers: "Which move should I try first in simulation?"

## Basic Features

All features are computed from a reference player perspective and are normalized
when possible.

### 1) Piece Difference

Function: piece_difference_feature(state, player)

- Measures material advantage.
- Formula: (player_pieces - opponent_pieces) / (player_pieces + opponent_pieces)
- Range: [-1, 1]

### 2) Mobility

Function: mobility_feature(state, player)

- Measures move availability advantage.
- Formula: (player_moves - opponent_moves) / (player_moves + opponent_moves)
- Range: [-1, 1]

### 3) Corner Control

Function: corner_control_feature(state, player)

- Uses dynamic corner positions from board size.
- Formula: (player_corners - opponent_corners) / total_corners
- Range: [-1, 1]

### 4) Edge Control

Function: edge_control_feature(state, player)

- Uses dynamic edge cells excluding corners.
- Formula: (player_edges - opponent_edges) / total_edge_cells_excluding_corners
- Range: [-1, 1]

### 5) Corner Proximity (Optional Light Feature)

Function: corner_proximity_feature(state, player)

- Looks at pieces adjacent to empty corners.
- Idea: adjacency to an empty corner is often risky/unstable.
- Positive values mean the opponent has more risky corner-adjacent pieces.

## Composite Evaluation Heuristics (Minimax)

Each evaluation follows a weighted sum:

Evaluate(state, player) = sum(w_i * feature_i(state, player))

### balanced

Function: evaluate_state_balanced(state, player)

Weights:

- piece: 0.30
- mobility: 0.25
- corners: 0.30
- edges: 0.10
- corner_proximity: 0.05

Use when you want a general-purpose, stable evaluation.

### aggressive

Function: evaluate_state_aggressive(state, player)

Weights:

- piece: 0.10
- mobility: 0.40
- corners: 0.35
- edges: 0.05
- corner_proximity: 0.10

Use when you want to prioritize tactical pressure (move options + corners).

### lightweight

Function: evaluate_state_lightweight(state, player)

Weights:

- piece: 0.65
- mobility: 0.35

Use when speed is more important than rich positional detail.

## Rollout Heuristics (MCTS)

Rollout heuristics are intentionally lightweight and shallow. They should guide
simulation, not replace tree search.

### rollout_simple

Core functions:

- score_move_rollout_simple(state, move, player)
- choose_rollout_move_simple(state, moves, player)

Priority order:

1. corners
2. mobility impact
3. edges

### rollout_successor_eval

Core functions:

- score_move_rollout_successor_eval(state, move, player, heuristic_name)
- choose_rollout_move_greedy(state, moves, player, heuristic_name)
- rank_moves_by_heuristic(state, moves, player, heuristic_name)

Approach:

- Apply each legal move once
- Evaluate successor with a lightweight state heuristic
- Choose highest-scoring move (or use ranking)

### rollout_topk

Core function:

- choose_rollout_move_topk(state, moves, player, k, heuristic_name, seed, rng)

Approach:

- Rank moves by heuristic
- Randomly sample among top-k moves
- Use seed for reproducibility

This keeps some exploration while still biasing toward better moves.

## Name-Based Access

### Evaluation resolver

Function: get_evaluation_function(name)

Accepted names:

- balanced
- aggressive
- lightweight

### Rollout policy resolver

Function: get_rollout_policy_function(name)

Accepted names:

- rollout_simple
- rollout_successor_eval
- rollout_topk

## Example of configuration-style usage

- Minimax can receive "balanced" and call get_evaluation_function("balanced")
- MCTS rollout can receive "rollout_topk" and call get_rollout_policy_function("rollout_topk")

This avoids hardcoding heuristic choices in agent logic and makes experiments
much easier to automate.
