import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, Instances
from PIL import Image


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp_box(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(_clamp(math.floor(x1), 0, width - 1))
    y1 = int(_clamp(math.floor(y1), 0, height - 1))
    x2 = int(_clamp(math.ceil(x2), x1 + 1, width))
    y2 = int(_clamp(math.ceil(y2), y1 + 1, height))
    return x1, y1, x2, y2


def _to_gray(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 2:
        return patch
    return 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]


def _texture_stats(patch: np.ndarray) -> Dict[str, float]:
    if patch.size == 0:
        return {}
    gray = _to_gray(patch.astype(np.float32))
    hist, _ = np.histogram(gray, bins=32, range=(0.0, 255.0), density=True)
    entropy = float(-np.sum(hist * np.log(hist + 1e-12)))
    return {
        "texture_mean": float(gray.mean()),
        "texture_std": float(gray.std()),
        "texture_entropy": entropy,
    }


def _percentile_range(values: List[float], low: float, high: float) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "min": float(np.percentile(arr, low)),
        "max": float(np.percentile(arr, high)),
    }


class PriorKnowledgeStatsBuilder:
    """
    Offline utility to build class-wise prior statistics from a registered dataset.
    """

    def __init__(self, low_percentile: float = 5.0, high_percentile: float = 95.0):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

    def build(self, dataset_name: str, output_path: str) -> None:
        if dataset_name not in DatasetCatalog.list():
            raise ValueError(f"Dataset {dataset_name} is not registered.")
        dataset = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        class_names = metadata.thing_classes

        stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        counts = defaultdict(int)

        for record in dataset:
            image = np.asarray(Image.open(record["file_name"]).convert("RGB"))
            height, width = image.shape[:2]
            for anno in record.get("annotations", []):
                if anno.get("iscrowd", 0):
                    continue
                cls_id = anno["category_id"]
                if cls_id >= len(class_names):
                    continue
                cls_name = class_names[cls_id]
                bbox = anno["bbox"]
                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h
                x1, y1, x2, y2 = _clamp_box((x1, y1, x2, y2), width, height)
                patch = image[y1:y2, x1:x2]
                feats = _texture_stats(patch)
                if not feats:
                    continue
                counts[cls_name] += 1
                for key, value in feats.items():
                    stats[cls_name][key].append(value)
                stats[cls_name]["aspect_ratio"].append(float((x2 - x1) / (y2 - y1 + 1e-6)))
                stats[cls_name]["area_ratio"].append(float(((x2 - x1) * (y2 - y1)) / (width * height + 1e-6)))
                stats[cls_name]["center_y"].append(float(((y1 + y2) / 2.0) / (height + 1e-6)))

        priors = {"meta": {"dataset": dataset_name, "counts": counts}}
        for cls_name, feature_map in stats.items():
            priors[cls_name] = {}
            for feat_name, values in feature_map.items():
                priors[cls_name][feat_name] = _percentile_range(values, self.low_percentile, self.high_percentile)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(priors, f, indent=2)


class PriorKnowledgeFilter:
    """
    Online pseudo-label filter that validates predictions against class-wise priors.
    """

    def __init__(
        self,
        prior_path: str,
        class_names: List[str],
        enabled: bool = True,
        high_vote: int = 2,
        medium_vote: int = 1,
        medium_score: float = 0.9,
        min_score: float = 0.7,
    ):
        self.enabled = enabled and bool(prior_path)
        self.class_names = class_names
        self.high_vote = high_vote
        self.medium_vote = medium_vote
        self.medium_score = medium_score
        self.min_score = min_score
        self.priors: Dict[str, Dict[str, Dict[str, float]]] = {}
        if self.enabled:
            with open(prior_path, "r") as f:
                self.priors = json.load(f)
            missing = [name for name in class_names if name not in self.priors]
            if missing:
                raise ValueError(f"Prior file {prior_path} missing classes: {missing}")

    def _vote(self, class_name: str, features: Dict[str, float]) -> int:
        prior = self.priors.get(class_name, {})
        votes = 0
        for key, bounds in prior.items():
            if key == "meta":
                continue
            value = features.get(key)
            if value is None:
                continue
            if bounds["min"] <= value <= bounds["max"]:
                votes += 1
            else:
                votes -= 1
        return votes

    def _patch_features(self, image: np.ndarray, box: Sequence[float]) -> Dict[str, float]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = _clamp_box(box, width, height)
        patch = image[y1:y2, x1:x2]
        feats = _texture_stats(patch)
        if not feats:
            feats = {}
        feats["aspect_ratio"] = float((x2 - x1) / (y2 - y1 + 1e-6))
        feats["area_ratio"] = float(((x2 - x1) * (y2 - y1)) / (width * height + 1e-6))
        feats["center_y"] = float(((y1 + y2) / 2.0) / (height + 1e-6))
        return feats

    def filter_instances(self, image_tensor: torch.Tensor, instances: Instances) -> Instances:
        if (not self.enabled) or len(instances) == 0:
            return instances
        np_image = image_tensor.detach().cpu().numpy()
        if np_image.shape[0] in (1, 3):
            np_image = np.transpose(np_image, (1, 2, 0))
        keep_indices: List[int] = []
        boxes_np = instances.pred_boxes.tensor.detach().cpu().numpy()
        classes = instances.pred_classes.detach().cpu().tolist()
        scores = instances.scores.detach().cpu().tolist()
        num_classes = len(self.class_names)
        for idx, (box, cls, score) in enumerate(zip(boxes_np, classes, scores)):
            if cls < 0 or cls >= num_classes:
                keep_indices.append(idx)
                continue
            class_name = self.class_names[cls]
            feats = self._patch_features(np_image, box)
            votes = self._vote(class_name, feats)
            if votes >= self.high_vote:
                keep_indices.append(idx)
            elif votes >= self.medium_vote and score >= self.medium_score:
                keep_indices.append(idx)
            elif votes >= 0 and score >= self.min_score:
                keep_indices.append(idx)
            # otherwise drop
        if len(keep_indices) == len(instances):
            return instances
        keep_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=instances.pred_boxes.tensor.device)
        filtered = Instances(instances.image_size)
        for field, value in instances.get_fields().items():
            filtered.set(field, value[keep_tensor])
        return filtered

    def filter_batch(self, batched_inputs: List[Dict], instances: List[Instances]) -> List[Instances]:
        if (not self.enabled) or not instances:
            return instances
        filtered = []
        for sample, inst in zip(batched_inputs, instances):
            filtered.append(self.filter_instances(sample["image"], inst))
        return filtered

