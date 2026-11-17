# Implementation update: learning-focused harsh ns-3 task-level scenario

## Design intent

This revision aims to improve DQN/PPO learnability while keeping LOCAL genuinely resource-constrained. It does not guarantee that learned policies beat GREEDY.

## Environment changes

- Primary harsh profile: 1.5 tasks/s/user; each user receives 3e9 requested CPU cycles/s against 0.5e9 CPU capacity. LOCAL remains six-times overloaded at each user.
- Edge battery is 1,500 J; user battery 1,000 J; radio TX/RX are 5/3 W.
- Main link budget calibrated to 802.11g 6 Mbps with log-distance exponent 2.8, 4 dB per-link static shadowing, and a 0.5 Mbps minimum bidirectional rate for a link to be eligible for offloading.
- Extreme overload profile preserves 2.5 tasks/s/user, exponent 3.2, and 6 dB shadowing.
- Infeasible actions are masked for memory, estimated current-task battery consumption, depleted batteries, and inadequate bidirectional link rate.
- All 60 C++ observation values are bounded in [0,1]. Queue and full-task upload-delay fields use log1p scaling with a 120-second cap. Python validates and clips only numerical overshoots; it does not double-normalize.

## RL changes

### DQN
- Dueling architecture.
- Double-DQN masked bootstrapping.
- 3-step return aggregation, including correct terminal flushing.
- Configurable train frequency (4 transitions), replay warm-up (1,200 transitions), batch 128, and target update (1,000 optimizer steps).
- Linear epsilon decay over 70,000 environment transitions, rather than over the requested episode count.

### PPO
- Separate actor and critic encoders.
- Orthogonal initialization.
- Normalized advantages, clipped policy objective, clipped Huber value objective, gradient clipping, target-KL early stopping.
- Eight epochs / minibatch 128; policy/value/KL/entropy/explained-variance diagnostics.

## Evaluation diagnostics

- Added DROP and CLOUD deterministic baselines.
- Retained LOCAL, NEAREST and objective-aligned GREEDY.
- Action counts and local/edge/cloud/DROP fractions are logged in both training and evaluation CSVs.
- Evaluation plot now includes objective, latency, energy, and deadline success.

## Files changed

- `ns3/uav-offloading.cc`
- `configs/harsh.yaml`, `configs/near_capacity.yaml`, `configs/moderate.yaml`; added `configs/extreme_overload.yaml`
- `src/observation.py` (new)
- `src/env/ns3_env.py`
- `src/agents/dqn.py`
- `src/agents/ppo.py`
- `src/baselines.py`
- `src/action_diagnostics.py` (new)
- `scripts/train.py`
- `scripts/evaluate.py`
- `tests/test_agents.py`, `tests/test_env_bridge.py`; added `tests/test_observation.py`
- `README.md`

## Verification in this build container

- `python -m compileall -q src scripts tests`: passed.
- `pytest -q`: 13 passed.
- Native C++ build against ns-3.49: not available in this container; the updated file must be built in the user's WSL ns-3.49 tree.
- No claim of packet-level UDP/MAC/energy fidelity is made.

## Retraining

All checkpoints from earlier observation semantics or network architectures are incompatible. Train new DQN/PPO checkpoints. Evaluate all policies on the same held-out seeds and preserve the full load-profile results, regardless of which policy wins.
