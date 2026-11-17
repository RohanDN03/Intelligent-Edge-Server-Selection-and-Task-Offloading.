# UAV Task Offloading with ns-3.49 + DQN/PPO

This is a separate experiment workspace. It does not modify the earlier custom-simulator project.

## What this project models

- 10 user UAVs, 3 edge UAVs, and one cloud node.
- No user-to-user offloading.
- Two equal subtasks per task.
- Five candidate destinations: source-local, Edge 0, Edge 1, Edge 2, and cloud.
- 15 unordered destination pairs (repetition allowed) plus DROP = 16 actions.
- 60 observations: 5 task/source fields + 5 candidates x 9 fields + 10 system fields.
- ns-3 supplies node/mobility objects, Wi-Fi setup, and `LogDistancePropagationLossModel`.
- Project-specific C++ task-level glue models task arrivals, CPU availability queues, memory reservations, battery accounting, propagation-based link-budget rates, and the synchronous Python bridge.

**Modeling boundary:** this is a propagation/link-budget and task-level simulation. It does not send UDP task payloads, measure actual MAC throughput/contention, or model UAV propulsion/flight energy. Do not describe results as packet-level throughput measurements.

## Objective (deadline excluded)

```text
J = 0.60 * (latency / latency_reference)
  + 0.20 * (system_energy / energy_reference)
  + 0.20 * (edge_load_imbalance / imbalance_reference)
reward = -J
```

Deadline is used only for success-rate reporting. DROP receives a fixed 10-second effective-latency penalty independent of the deadline. System energy includes CPU energy plus TX/RX energy at both endpoints of offloaded links.

## Calibrated but still harsh workload profiles

`configs/harsh.yaml` is the recommended primary profile:

- 1.5 task arrivals/s/user (about 900 tasks per 60-second episode).
- 2e9 CPU cycles/task, 1 MB input, 0.1 MB output, 128 MB memory/task.
- User CPU 0.5 GHz, edge CPU 2 GHz each, cloud CPU 20 GHz.
- User battery 1,000 J, edge battery 1,500 J; low radio capacity and 5 W/3 W TX/RX energy.
- 802.11g at 6 Mbps; log-distance path-loss exponent 2.8, 4 dB static per-link shadowing, and a 0.5 Mbps minimum usable bidirectional link rate.
- Deadline 2 seconds (reporting only).

LOCAL remains deliberately stressed: each user receives approximately 1.5 x 2e9 = 3e9 CPU cycles/s, while its local CPU can process only 0.5e9 cycles/s. That is six times its local CPU capacity. Offloading opportunities are retained, but wireless links are still constrained by 6 Mbps PHY/bandwidth, propagation loss, shadowing, radio energy, finite batteries, and queues.

Other profiles:

| Config | Arrival rate/user | Mean compute demand | Purpose |
|---|---:|---:|---|
| `moderate.yaml` | 1.0/s | 20 GHz | Lower-load sensitivity |
| `near_capacity.yaml` | 1.35/s | 27 GHz | High load with some aggregate headroom |
| `harsh.yaml` | 1.5/s | 30 GHz | Primary harsh-local experiment; 31 GHz nominal aggregate CPU |
| `extreme_overload.yaml` | 2.5/s | 50 GHz | Deliberately overloaded stress test |

The aggregate CPU totals are nominal capacity only; they do not guarantee stable operation once link limits, node-specific queues, and task placement are considered. Keep all profiles in the experiment rather than selecting only the one that favours a policy.

## Changes intended to improve learning without weakening LOCAL or baselines

### Environment and observations

- Retains event-time CPU queues and memory reservations; the fixed 0.05-second queue decrement is not reintroduced.
- Feasibility masks reject depleted/insufficient-memory destinations and remote paths below the configured minimum bidirectional link rate. They also reject actions that cannot pay the estimated current-task energy.
- The main radio profile uses a 2.8 path-loss exponent and 4 dB shadowing; the separate extreme profile keeps substantially harsher propagation.
- Observation delay fields use `log1p(seconds) / log1p(120)` and all features are bounded in [0, 1]. The same transform is used in training/evaluation; the greedy baseline decodes it back to seconds.
- Battery, memory, distance, rate, CPU and queue information remain visible to the agents.

### DQN

- Dueling Double-DQN with 3-step returns.
- Masked action selection and masked next-action selection.
- Replay warm-up, batch size, target update interval, train frequency, gradient clipping, and epsilon schedule are config-driven.
- Epsilon decays over environment transitions, not over the user-requested episode count. Five-episode smoke runs therefore do not prematurely force epsilon to its minimum.

### PPO

- Separate actor and critic encoders reduce shared-feature gradient interference.
- Orthogonal initialization, normalized advantages, clipped policy objective, clipped Huber value loss, gradient clipping, and target-KL early stopping.
- Eight update epochs with minibatches of 128 in the primary profile.
- Reports policy/value loss, entropy, KL, clip fraction, explained variance, update count and early stopping.

### Baselines and diagnostics

Evaluation includes DROP, LOCAL, CLOUD, NEAREST, and GREEDY in addition to DQN/PPO. GREEDY remains a strong one-step objective-aligned heuristic; it is not artificially weakened. Training/evaluation CSVs include action counts and local/edge/cloud/DROP fractions so we can see what each algorithm actually chooses.

The objective weights and deadline treatment are unchanged. These changes are intended to create a more learnable, properly scaled, high-load problem; they do not guarantee DQN/PPO will beat GREEDY.

## 1. Build ns-3.49

If ns-3.49 is already built and its test suite passed, do not clone it again.

```bash
cd ~/workspace/ns-3.49
cp /path/to/ns3_uav_offloading/ns3/uav-offloading.cc scratch/uav-offloading.cc
./ns3 build

export NS3_UAV_BINARY="$HOME/workspace/ns-3.49/build/scratch/ns3.49-uav-offloading-default"
```

If the executable path differs:

```bash
find "$HOME/workspace/ns-3.49/build" -type f -executable -iname '*uav-offloading*' -print
```

## 2. Install Python dependencies and verify integration

```bash
cd /path/to/ns3_uav_offloading
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export NS3_UAV_BINARY="$HOME/workspace/ns-3.49/build/scratch/ns3.49-uav-offloading-default"
python scripts/check_backend.py
pytest -q
```

The backend check performs a one-task READY/OBS/MASK/ACTION/STEP/SUMMARY round trip and checks that all 60 observation values are finite and bounded.

## 3. Train from scratch

Observation version 3 and the DQN/PPO network architectures differ from earlier releases. **Do not reuse older checkpoints.** The commands below create fresh checkpoints.

Recommended initial check (10 episodes):

```bash
python scripts/train.py --algo dqn --episodes 10 --seed 7 --config configs/harsh.yaml --run-name harsh_seed7
python scripts/train.py --algo ppo --episodes 10 --seed 7 --config configs/harsh.yaml --run-name harsh_seed7
```

Then use at least 100 episodes after confirming learning curves and action distributions:

```bash
python scripts/train.py --algo dqn --episodes 100 --seed 7 --config configs/harsh.yaml --run-name harsh_seed7_100ep
python scripts/train.py --algo ppo --episodes 100 --seed 7 --config configs/harsh.yaml --run-name harsh_seed7_100ep
```

Checkpoint paths are `checkpoints/dqn_<run-name>_latest.pt` and `checkpoints/ppo_<run-name>_latest.pt` when `--run-name` is supplied. Training logs and metadata use matching names in `results/`.

## 4. Evaluate on common held-out seeds

```bash
python scripts/evaluate.py --episodes 30 --seed 1000 --config configs/harsh.yaml \
  --dqn checkpoints/dqn_harsh_seed7_100ep_latest.pt \
  --ppo checkpoints/ppo_harsh_seed7_100ep_latest.pt \
  --run-name harsh_1000
```

Repeat evaluation against all load profiles with the same checkpoints to measure generalization:

```bash
python scripts/evaluate.py --episodes 30 --seed 2000 --config configs/moderate.yaml --dqn checkpoints/dqn_harsh_seed7_100ep_latest.pt --ppo checkpoints/ppo_harsh_seed7_100ep_latest.pt --run-name moderate
python scripts/evaluate.py --episodes 30 --seed 3000 --config configs/near_capacity.yaml --dqn checkpoints/dqn_harsh_seed7_100ep_latest.pt --ppo checkpoints/ppo_harsh_seed7_100ep_latest.pt --run-name near_capacity
python scripts/evaluate.py --episodes 30 --seed 4000 --config configs/extreme_overload.yaml --dqn checkpoints/dqn_harsh_seed7_100ep_latest.pt --ppo checkpoints/ppo_harsh_seed7_100ep_latest.pt --run-name extreme_overload
```

Outputs include `evaluation_episodes*.csv`, `evaluation_summary*.csv`, and `policy_comparison*.png`. Per-action counts, DROP rate, and destination-subtask fractions are included.

## Experiment discipline

1. Compile the new C++ file in the user's ns-3.49 tree and run the backend smoke check.
2. Start with 10 episodes to check action distributions and loss stability; do not infer performance from smoke tests.
3. Train on training seeds only. Use separate validation seeds for checkpoint selection, and held-out seeds for final reporting.
4. Compare DQN/PPO with DROP, LOCAL, CLOUD, NEAREST and GREEDY on identical seeds.
5. Report means and standard deviations for objective, latency, energy, deadline success, DROP rate, action selection, battery and resource outcomes.
6. Keep results from moderate, harsh, and extreme-overload regimes. Never force an RL win by weakening a baseline or changing only test conditions.

## Known limitations

- Constant-position nodes; no UAV flight paths, propulsion energy, or handovers.
- No UDP task payloads, packet-delivery measurements, MAC contention/queue traces, or ns-3 energy-source state transitions.
- The propagation-derived effective rate is a link-budget estimate, not measured PHY/MAC throughput.
- Cloud is currently represented as a direct peer in this task-level topology, not a separate edge-to-cloud backhaul.

Before publication, calibrate the link-budget model against a defined UAV communication scenario and clearly describe these limitations. A later higher-fidelity version should use ns-3 Internet/Applications, UDP sockets, Wi-Fi tracing, and energy sources.
