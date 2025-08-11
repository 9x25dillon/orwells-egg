from typing import Dict, Any


def coach_update(metrics: Dict[str, float], entropy_report: Dict[str, float], state: Dict[str, Any]) -> Dict[str, Any]:
    # shrink lr if dev loss stagnates
    if metrics.get("dev_loss_delta", 0.0) > -1e-3:
        state["lr"] = max(state.get("lr", 1e-3) * 0.8, 1e-6)
    # increase top_k if entropy is low
    if entropy_report.get("avg_token_entropy", 99.0) < state.get("entropy_floor", 3.0):
        state["top_k"] = min(state.get("top_k", 50) + 10, 200)
    return state