from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List


@dataclass
class Packet:
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    hydration: float = 0.0  # cache warmth / preprocessing richness
    pigment: float = 0.0    # provenance/sensitivity 0..1


Layer = Callable[[Packet], Packet]


# --- Strata ---

def stratum_corneum(p: Packet) -> Packet:
    bad = ["drop table", "<script>", "onerror=", "file:///", "../../../"]
    if any(x in p.content.lower() for x in bad):
        p.meta["rejected"] = "surface-guard"
        return p
    p.hydration += 0.1
    return p


def stratum_granulosum(p: Packet) -> Packet:
    # pack minimal features that downstream always needs
    p.meta["features"] = {"len": len(p.content), "hash": hash(p.content)}
    p.hydration += 0.2
    return p


def stratum_spinosum(p: Packet) -> Packet:
    # tighten junctions: link/provenance checks
    p.meta["links_ok"] = True
    p.pigment = min(1.0, p.pigment + 0.2)
    return p


def stratum_basale(p: Packet) -> Packet:
    # stem layer: may spawn enrich/repair jobs
    if p.hydration < 0.5:
        p.meta.setdefault("spawn", []).append("enrich")
    return p


DEFAULT_PIPELINE: List[Layer] = [
    stratum_corneum,
    stratum_granulosum,
    stratum_spinosum,
    stratum_basale,
]



def run_skin(packet: Packet, layers: List[Layer] | None = None) -> Packet:
    layers = layers or DEFAULT_PIPELINE
    for layer in layers:
        packet = layer(packet)
        if packet.meta.get("rejected"):
            break
    return packet
