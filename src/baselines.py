import math

from .action_space import PAIRS, DROP_ACTION
from .observation import decode_delay_feature


def local(mask):
    return 0 if mask[0] else DROP_ACTION


def drop(mask):
    return DROP_ACTION


def cloud(mask):
    action = PAIRS.index((4, 4))
    return action if mask[action] else DROP_ACTION


def nearest(obs, mask):
    """Choose the closest feasible edge for both subtasks; exclude LOCAL."""
    choices = []
    for candidate in (1, 2, 3):
        action = PAIRS.index((candidate, candidate))
        if mask[action]:
            distance = float(obs[5 + candidate * 9 + 4])
            choices.append((distance, action))
    return min(choices)[1] if choices else (0 if mask[0] else DROP_ACTION)


def greedy(obs, mask, cfg):
    """One-step immediate-objective baseline using decoded bounded features."""
    weights = cfg["objective_weights"]
    latency_ref = float(cfg["latency_reference_s"])
    energy_ref = float(cfg["energy_reference_j"])
    imbalance_ref = float(cfg["imbalance_reference"])
    drop_latency = float(cfg["drop_latency_s"])
    cycles_per_subtask = float(cfg["cpu_cycles_per_task"]) / 2.0
    cpu_coeff = float(cfg["cpu_energy_coefficient"])
    radio_tx_w = float(cfg["radio_tx_w"])
    radio_rx_w = float(cfg["radio_rx_w"])
    input_mb = float(cfg["input_size_mb"])
    output_mb = float(cfg["output_size_mb"])

    edge_queues_initial = [decode_delay_feature(float(obs[50 + i])) for i in range(3)]
    best_cost = math.inf
    best_action = DROP_ACTION

    for action, (first, second) in enumerate(PAIRS):
        if not bool(mask[action]):
            continue
        cpu_available = {}
        edge_queues = edge_queues_initial.copy()
        finish_latency = 0.0
        total_energy = 0.05

        for candidate in (first, second):
            base = 5 + candidate * 9
            cpu_norm = max(float(obs[base]), 1e-8)
            cpu_hz = cpu_norm * float(cfg["cloud_cpu_hz"])
            initial_wait = decode_delay_feature(float(obs[base + 1]))
            full_input_delay = decode_delay_feature(float(obs[base + 5]))

            if candidate == 0:
                uplink = 0.0
                downlink = 0.0
            else:
                uplink = 0.5 * full_input_delay
                downlink = (0.5 * output_mb / max(input_mb, 1e-12)) * full_input_delay

            prior_available = cpu_available.get(candidate, initial_wait)
            service_start = max(uplink, prior_available)
            compute = cycles_per_subtask / cpu_hz
            service_end = service_start + compute
            completion = service_end + downlink
            finish_latency = max(finish_latency, completion)
            cpu_available[candidate] = service_end

            cpu_energy = cpu_coeff * cycles_per_subtask * cpu_hz * cpu_hz
            radio_energy = 0.0 if candidate == 0 else (radio_tx_w + radio_rx_w) * (uplink + downlink)
            total_energy += cpu_energy + radio_energy
            if candidate in (1, 2, 3):
                edge_queues[candidate - 1] = max(edge_queues[candidate - 1], service_end)

        latency = finish_latency + 0.02
        edge_mean = sum(edge_queues) / 3.0
        edge_var = sum((q - edge_mean) ** 2 for q in edge_queues) / 3.0
        imbalance = min(1.0, math.sqrt(edge_var) / (edge_mean + 0.1))
        cost = (
            float(weights["latency"]) * latency / latency_ref
            + float(weights["energy"]) * total_energy / energy_ref
            + float(weights["imbalance"]) * imbalance / imbalance_ref
        )
        if cost < best_cost:
            best_cost, best_action = cost, action

    if bool(mask[DROP_ACTION]):
        drop_cost = (
            float(weights["latency"]) * drop_latency / latency_ref
            + float(weights["energy"]) * 0.05 / energy_ref
        )
        if drop_cost < best_cost:
            best_action = DROP_ACTION
    return best_action
