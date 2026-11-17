from itertools import combinations_with_replacement

DESTINATIONS = ("LOCAL", "EDGE0", "EDGE1", "EDGE2", "CLOUD")
PAIRS = tuple(combinations_with_replacement(range(len(DESTINATIONS)), 2))
DROP_ACTION = len(PAIRS)
ACTION_DIM = DROP_ACTION + 1
assert len(PAIRS) == 15 and ACTION_DIM == 16

def decode_action(action: int):
    if action == DROP_ACTION:
        return None
    if not 0 <= action < len(PAIRS):
        raise ValueError(f"Invalid action {action}")
    return tuple(DESTINATIONS[i] for i in PAIRS[action])
