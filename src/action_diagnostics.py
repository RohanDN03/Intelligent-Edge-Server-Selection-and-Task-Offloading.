"""Per-action and per-destination diagnostics for RL and baseline policies."""
from .action_space import PAIRS, DROP_ACTION


def summarize_actions(action_counts):
    counts = [int(x) for x in action_counts]
    tasks = sum(counts)
    dropped = counts[DROP_ACTION]
    placed_subtasks = 2 * (tasks - dropped)
    local_count = edge_count = cloud_count = 0
    for action, count in enumerate(counts[:DROP_ACTION]):
        first, second = PAIRS[action]
        for destination in (first, second):
            if destination == 0:
                local_count += count
            elif destination == 4:
                cloud_count += count
            else:
                edge_count += count
    denom = max(1, placed_subtasks)
    output = {
        "drop_rate": dropped / max(1, tasks),
        "local_subtask_fraction": local_count / denom,
        "edge_subtask_fraction": edge_count / denom,
        "cloud_subtask_fraction": cloud_count / denom,
    }
    output.update({f"action_{i:02d}_count": counts[i] for i in range(len(counts))})
    return output
