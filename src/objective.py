from dataclasses import dataclass

@dataclass(frozen=True)
class ObjectiveConfig:
    latency_ref_s: float = 1.0
    energy_ref_j: float = 10.0
    imbalance_ref: float = 1.0
    latency_weight: float = 0.60
    energy_weight: float = 0.20
    imbalance_weight: float = 0.20

    def validate(self):
        weights = (self.latency_weight, self.energy_weight, self.imbalance_weight)
        if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("Objective weights must be non-negative and sum to 1")
        if min(self.latency_ref_s, self.energy_ref_j, self.imbalance_ref) <= 0:
            raise ValueError("Objective reference scales must be positive")

def objective(latency_s: float, energy_j: float, imbalance: float,
              cfg: ObjectiveConfig = ObjectiveConfig()) -> float:
    """No deadline term by design; deadline violations are evaluation-only."""
    cfg.validate()
    return (cfg.latency_weight * latency_s / cfg.latency_ref_s
            + cfg.energy_weight * energy_j / cfg.energy_ref_j
            + cfg.imbalance_weight * imbalance / cfg.imbalance_ref)
