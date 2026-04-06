"""Evaluate YOLO26 segmentation models on COCO with bbox and mask metrics."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PKGS = REPO_ROOT / ".local_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common import (
    HailoSegmentationInferenceEngine,
    infer_segmentation_config_from_hef,
    preprocess_rgb_image,
    scale_detections_to_original,
)


def _load_rgb_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _encode_binary_mask(mask: np.ndarray) -> Dict[str, object]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _project_masks_to_original(
    masks: Optional[np.ndarray],
    orig_h: int,
    orig_w: int,
    scale: float,
    pad_w: int,
    pad_h: int,
    target_size: int = 640,
) -> List[np.ndarray]:
    if masks is None:
        return []

    projected = []
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)

    for mask in masks:
        mask_target = cv2.resize(mask.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask_crop = mask_target[pad_h : pad_h + new_h, pad_w : pad_w + new_w]
        mask_orig = cv2.resize(mask_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        projected.append((mask_orig > 0.5).astype(np.uint8))

    return projected


def _summarize_latency(samples: Sequence[float]) -> Dict[str, Optional[float]]:
    if not samples:
        return {"mean_ms": None, "p50_ms": None, "p90_ms": None, "p95_ms": None}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean() * 1000.0),
        "p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "p90_ms": float(np.percentile(arr, 90) * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
    }


def _evaluate_predictions(coco_gt: COCO, predictions: List[Dict], iou_type: str) -> Dict[str, float]:
    coco_dt = coco_gt.loadRes(predictions) if predictions else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.imgIds = sorted(set(pred["image_id"] for pred in predictions)) if predictions else []
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR1": float(stats[6]),
        "AR10": float(stats[7]),
        "AR100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }


def _build_coco_mappings(coco_gt: COCO) -> Tuple[Dict[str, int], Dict[str, int]]:
    categories = coco_gt.dataset["categories"]
    name_to_cat_id = {cat["name"]: int(cat["id"]) for cat in categories}
    file_to_image_id = {img["file_name"]: int(img["id"]) for img in coco_gt.dataset["images"]}
    return name_to_cat_id, file_to_image_id


def _prediction_record(
    image_id: int,
    category_id: int,
    box_xyxy: Sequence[float],
    score: float,
    binary_mask: np.ndarray,
) -> Dict[str, object]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x1, y1, width, height],
        "score": float(score),
        "segmentation": _encode_binary_mask(binary_mask),
    }


def run_hailo_eval(args: argparse.Namespace) -> Dict[str, object]:
    inferred = infer_segmentation_config_from_hef(args.hef)
    num_classes = args.num_classes or inferred["num_classes"]
    num_masks = args.num_masks or inferred["num_masks"]
    class_names = args.class_names or inferred["class_names"]
    if num_classes is None or num_masks is None:
        raise ValueError("Unable to infer segmentation config from HEF.")

    coco_gt = COCO(args.annotations)
    name_to_cat_id, file_to_image_id = _build_coco_mappings(coco_gt)

    engine = HailoSegmentationInferenceEngine(
        hef_path=args.hef,
        num_classes=num_classes,
        num_masks=num_masks,
        class_names=class_names,
    )

    bbox_predictions: List[Dict[str, object]] = []
    segm_predictions: List[Dict[str, object]] = []
    total_times: List[float] = []
    hailo_times: List[float] = []
    post_times: List[float] = []
    preprocess_times: List[float] = []

    image_paths = sorted(Path(args.images_dir).glob("*.jpg"))
    if args.limit:
        image_paths = image_paths[: args.limit]

    try:
        for idx, image_path in enumerate(image_paths, start=1):
            image_id = file_to_image_id.get(image_path.name)
            if image_id is None:
                continue

            image_rgb = _load_rgb_image(image_path)
            orig_h, orig_w = image_rgb.shape[:2]

            t_pre = time.perf_counter()
            input_tensor, _, scale, pad_w, pad_h = preprocess_rgb_image(image_rgb, normalize=args.normalize)
            preprocess_times.append(time.perf_counter() - t_pre)

            detections, masks, stats = engine.infer(
                input_tensor,
                verbose=False,
                conf_threshold=args.conf_threshold,
                mask_threshold=args.mask_threshold,
                iou_threshold=args.iou_threshold,
            )
            total_times.append(stats.total_time)
            hailo_times.append(stats.hailo_inference_time)
            post_times.append(stats.postprocess_time)

            detections = scale_detections_to_original(detections, orig_h, orig_w, scale, pad_w, pad_h)
            projected_masks = _project_masks_to_original(masks, orig_h, orig_w, scale, pad_w, pad_h)

            for det, mask in zip(detections, projected_masks):
                cls_name = det["cls_name"]
                if cls_name not in name_to_cat_id:
                    continue
                category_id = name_to_cat_id[cls_name]
                record = _prediction_record(
                    image_id=image_id,
                    category_id=category_id,
                    box_xyxy=[det["x1"], det["y1"], det["x2"], det["y2"]],
                    score=det["conf"],
                    binary_mask=mask,
                )
                bbox_record = dict(record)
                bbox_record.pop("segmentation", None)
                bbox_predictions.append(bbox_record)
                segm_predictions.append(record)

            if idx % args.log_every == 0:
                print(f"[hailo] processed {idx}/{len(image_paths)} images")
    finally:
        engine.close()

    bbox_metrics = _evaluate_predictions(coco_gt, bbox_predictions, "bbox")
    segm_metrics = _evaluate_predictions(coco_gt, segm_predictions, "segm")

    total_time_sum = float(sum(total_times) + sum(preprocess_times))
    summary = {
        "mode": "hailo",
        "images": len(image_paths),
        "predictions": len(segm_predictions),
        "fps": float(len(image_paths) / total_time_sum) if total_time_sum > 0 else None,
        "latency": {
            "preprocess": _summarize_latency(preprocess_times),
            "hailo": _summarize_latency(hailo_times),
            "postprocess": _summarize_latency(post_times),
            "total": _summarize_latency([a + b for a, b in zip(preprocess_times, total_times)]),
        },
        "bbox_metrics": bbox_metrics,
        "segm_metrics": segm_metrics,
    }
    return {"summary": summary, "bbox_predictions": bbox_predictions, "segm_predictions": segm_predictions}


def run_cpu_eval(args: argparse.Namespace) -> Dict[str, object]:
    from ultralytics import YOLO

    coco_gt = COCO(args.annotations)
    name_to_cat_id, file_to_image_id = _build_coco_mappings(coco_gt)
    model = YOLO(args.weights)

    bbox_predictions: List[Dict[str, object]] = []
    segm_predictions: List[Dict[str, object]] = []
    total_times: List[float] = []
    image_paths = sorted(Path(args.images_dir).glob("*.jpg"))
    if args.limit:
        image_paths = image_paths[: args.limit]

    for idx, image_path in enumerate(image_paths, start=1):
        image_id = file_to_image_id.get(image_path.name)
        if image_id is None:
            continue

        t0 = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            device="cpu",
            imgsz=args.imgsz,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            max_det=args.max_det,
            verbose=False,
            save=False,
            retina_masks=True,
        )[0]
        total_times.append(time.perf_counter() - t0)

        if result.boxes is None or result.masks is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy()

        if masks.ndim == 3 and masks.shape[1:] != result.orig_shape:
            resized_masks = []
            orig_h, orig_w = result.orig_shape
            for mask in masks:
                resized_masks.append(
                    cv2.resize(mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5
                )
            masks = np.asarray(resized_masks, dtype=np.uint8)
        else:
            masks = (masks > 0.5).astype(np.uint8)

        for box, score, cls_id, mask in zip(boxes, scores, classes, masks):
            cls_name = model.names[int(cls_id)]
            category_id = name_to_cat_id.get(cls_name)
            if category_id is None:
                continue
            record = _prediction_record(
                image_id=image_id,
                category_id=category_id,
                box_xyxy=box.tolist(),
                score=float(score),
                binary_mask=mask.astype(np.uint8),
            )
            bbox_record = dict(record)
            bbox_record.pop("segmentation", None)
            bbox_predictions.append(bbox_record)
            segm_predictions.append(record)

        if idx % args.log_every == 0:
            print(f"[cpu] processed {idx}/{len(image_paths)} images")

    bbox_metrics = _evaluate_predictions(coco_gt, bbox_predictions, "bbox")
    segm_metrics = _evaluate_predictions(coco_gt, segm_predictions, "segm")
    total_time_sum = float(sum(total_times))
    summary = {
        "mode": "cpu",
        "images": len(image_paths),
        "predictions": len(segm_predictions),
        "fps": float(len(image_paths) / total_time_sum) if total_time_sum > 0 else None,
        "latency": {"total": _summarize_latency(total_times)},
        "bbox_metrics": bbox_metrics,
        "segm_metrics": segm_metrics,
    }
    return {"summary": summary, "bbox_predictions": bbox_predictions, "segm_predictions": segm_predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation models on COCO.")
    parser.add_argument("--mode", choices=["cpu", "hailo"], required=True)
    parser.add_argument("--images-dir", required=True, help="Directory containing COCO val images")
    parser.add_argument("--annotations", required=True, help="Path to COCO instances_val2017.json")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")
    parser.add_argument("--predictions-json", help="Optional path to save segmentation predictions JSON")
    parser.add_argument("--limit", type=int, help="Optional image limit for smoke tests")
    parser.add_argument("--log-every", type=int, default=100, help="Progress log interval")
    parser.add_argument("--conf-threshold", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--weights", help="Path to FP32 segmentation weights for CPU mode")
    parser.add_argument("--hef", help="Path to HEF for Hailo mode")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--num-masks", type=int)
    parser.add_argument("--class-names", nargs="+")
    args = parser.parse_args()

    if args.mode == "cpu" and not args.weights:
        parser.error("--weights is required for --mode cpu")
    if args.mode == "hailo" and not args.hef:
        parser.error("--hef is required for --mode hailo")

    result = run_cpu_eval(args) if args.mode == "cpu" else run_hailo_eval(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")

    if args.predictions_json:
        preds_path = Path(args.predictions_json)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_path.write_text(json.dumps(result["segm_predictions"]), encoding="utf-8")


if __name__ == "__main__":
    main()
