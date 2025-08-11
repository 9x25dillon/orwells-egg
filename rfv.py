import torch
from typing import Any, Dict, Optional


def make_rfv_snapshot(model: torch.nn.Module, sample_batch: torch.Tensor, label_map: Optional[Dict[str, Any]] = None):
    model.eval()
    with torch.no_grad():
        features = model(sample_batch).detach().cpu()
    meta = {"arch": "torch", "shape": list(features.shape), "labels": label_map}
    # In practice, store features to an object store and return that URI
    return {"rdata_uri": "s3://bucket/rfv/xyz.pt", "rml_meta": meta}