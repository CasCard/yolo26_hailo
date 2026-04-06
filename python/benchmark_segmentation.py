"""Benchmark a segmentation HEF on Raspberry Pi + Hailo-8L."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from common import HailoSegmentationInferenceEngine, infer_segmentation_config_from_hef, preprocess_rgb_image


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None


def _read_cpu_temp_c() -> Optional[float]:
    for candidate in ["/sys/class/thermal/thermal_zone0/temp", "/sys/devices/virtual/thermal/thermal_zone0/temp"]:
        raw = _read_text(candidate)
        if raw:
            try:
                value = float(raw)
                return value / 1000.0 if value > 1000 else value
            except ValueError:
                pass
    return None


def _read_cpu_freq_mhz() -> Optional[float]:
    raw = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if not raw:
        return None
    try:
        return float(raw) / 1000.0
    except ValueError:
        return None


def _read_rss_mb() -> Optional[float]:
    status_text = _read_text("/proc/self/status")
    if not status_text:
        return None
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1]) / 1024.0
                except ValueError:
                    return None
    return None


def _get_platform_info() -> Dict[str, Optional[str]]:
    return {
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "machine": os.uname().machine,
        "model": _read_text("/proc/device-tree/model") or _read_text("/sys/firmware/devicetree/base/model"),
    }


class ProcessCpuMeter:
    def __init__(self) -> None:
        self._last_wall = None
        self._last_cpu = None

    def reset(self) -> None:
        self._last_wall = time.perf_counter()
        self._last_cpu = time.process_time()

    def sample_percent(self) -> Optional[float]:
        current_wall = time.perf_counter()
        current_cpu = time.process_time()
        if self._last_wall is None or self._last_cpu is None:
            self._last_wall = current_wall
            self._last_cpu = current_cpu
            return None
        wall_delta = current_wall - self._last_wall
        cpu_delta = current_cpu - self._last_cpu
        self._last_wall = current_wall
        self._last_cpu = current_cpu
        if wall_delta <= 0:
            return None
        return 100.0 * cpu_delta / wall_delta


def _find_images(images_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(sorted(images_dir.glob(pattern)))
    return image_paths


def _load_rgb_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _prepare_cache(image_paths: List[Path], cache_images: int) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    for path in image_paths[: max(0, cache_images)]:
        cache[str(path)] = _load_rgb_image(path)
    return cache


def _compute_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "min": None, "p50": None, "p90": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _summarize_iterations(records: List[Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        "preprocess_ms": _compute_stats([r["preprocess_ms"] for r in records]),
        "hailo_ms": _compute_stats([r["hailo_ms"] for r in records]),
        "postprocess_ms": _compute_stats([r["postprocess_ms"] for r in records]),
        "total_ms": _compute_stats([r["total_ms"] for r in records]),
        "fps": _compute_stats([r["fps"] for r in records]),
        "detections_per_frame": _compute_stats([r["detections"] for r in records]),
        "process_cpu_percent": _compute_stats([r["process_cpu_percent"] for r in records if r["process_cpu_percent"] is not None]),
        "rss_mb": _compute_stats([r["rss_mb"] for r in records if r["rss_mb"] is not None]),
        "cpu_temp_c": _compute_stats([r["cpu_temp_c"] for r in records if r["cpu_temp_c"] is not None]),
        "cpu_freq_mhz": _compute_stats([r["cpu_freq_mhz"] for r in records if r["cpu_freq_mhz"] is not None]),
    }


def benchmark(args: argparse.Namespace) -> Dict:
    inferred = infer_segmentation_config_from_hef(args.hef)
    resolved_num_classes = args.num_classes or inferred["num_classes"]
    resolved_num_masks = args.num_masks or inferred["num_masks"]
    resolved_class_names = args.class_names or inferred["class_names"]
    if resolved_num_classes is None or resolved_num_masks is None:
        raise ValueError("Unable to infer class or mask count from the HEF.")
    if resolved_class_names is None:
        resolved_class_names = [f"class_{i}" for i in range(resolved_num_classes)]

    source_mode = "single_image" if args.image else "image_directory"
    cached_image = _load_rgb_image(Path(args.image)) if args.image else None
    image_paths = _find_images(Path(args.images_dir)) if args.images_dir else []
    image_cache = _prepare_cache(image_paths, args.cache_images)

    engine = HailoSegmentationInferenceEngine(
        hef_path=args.hef,
        num_classes=resolved_num_classes,
        num_masks=resolved_num_masks,
        class_names=resolved_class_names,
    )
    cpu_meter = ProcessCpuMeter()
    records: List[Dict] = []

    try:
        cpu_meter.reset()
        for i in range(args.warmup):
            if source_mode == "single_image":
                input_data, _, _, _, _ = preprocess_rgb_image(cached_image, normalize=args.normalize)
            else:
                image_path = image_paths[i % len(image_paths)]
                image_rgb = image_cache.get(str(image_path))
                if image_rgb is None:
                    image_rgb = _load_rgb_image(image_path)
                input_data, _, _, _, _ = preprocess_rgb_image(image_rgb, normalize=args.normalize)
            engine.infer(
                input_data,
                verbose=False,
                conf_threshold=args.conf_threshold,
                mask_threshold=args.mask_threshold,
                iou_threshold=args.iou_threshold,
            )

        cpu_meter.reset()
        for i in range(args.iterations):
            t_pre = time.perf_counter()
            if source_mode == "single_image":
                input_data, _, _, _, _ = preprocess_rgb_image(cached_image, normalize=args.normalize)
                source_name = "cached_single_image"
            else:
                image_path = image_paths[i % len(image_paths)]
                source_name = str(image_path)
                image_rgb = image_cache.get(source_name)
                if image_rgb is None:
                    image_rgb = _load_rgb_image(image_path)
                input_data, _, _, _, _ = preprocess_rgb_image(image_rgb, normalize=args.normalize)
            preprocess_time = time.perf_counter() - t_pre

            detections, masks, stats = engine.infer(
                input_data,
                verbose=False,
                conf_threshold=args.conf_threshold,
                mask_threshold=args.mask_threshold,
                iou_threshold=args.iou_threshold,
            )

            total_time = preprocess_time + stats.total_time
            records.append(
                {
                    "iteration": i + 1,
                    "source": source_name,
                    "preprocess_ms": preprocess_time * 1000.0,
                    "hailo_ms": stats.hailo_inference_time * 1000.0,
                    "postprocess_ms": stats.postprocess_time * 1000.0,
                    "total_ms": total_time * 1000.0,
                    "fps": 1.0 / total_time if total_time > 0 else None,
                    "detections": len(detections),
                    "masks": int(masks.shape[0]) if masks is not None else 0,
                    "process_cpu_percent": cpu_meter.sample_percent(),
                    "rss_mb": _read_rss_mb(),
                    "cpu_temp_c": _read_cpu_temp_c(),
                    "cpu_freq_mhz": _read_cpu_freq_mhz(),
                }
            )
    finally:
        engine.close()

    summary = _summarize_iterations(records)
    output = {
        "platform": _get_platform_info(),
        "benchmark": {
            "hef": str(args.hef),
            "network_group_name": inferred["network_group_name"],
            "iterations": args.iterations,
            "warmup": args.warmup,
            "source_mode": source_mode,
            "image": args.image,
            "images_dir": args.images_dir,
            "cache_images": args.cache_images,
            "num_classes": resolved_num_classes,
            "class_names": resolved_class_names,
            "num_masks": resolved_num_masks,
            "conf_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "mask_threshold": args.mask_threshold,
            "normalize": args.normalize,
        },
        "summary": summary,
        "iterations": records,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a segmentation HEF on Raspberry Pi + Hailo-8L")
    parser.add_argument("--hef", type=str, required=True, help="Path to segmentation HEF")
    parser.add_argument("--image", type=str, help="Single image to benchmark repeatedly")
    parser.add_argument("--images-dir", type=str, help="Directory of images for multi-image benchmark")
    parser.add_argument("--iterations", type=int, default=200, help="Measured iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--cache-images", type=int, default=0, help="Number of directory images to cache in RAM")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold")
    parser.add_argument("--num-classes", type=int, help="Number of segmentation classes (auto-inferred if omitted)")
    parser.add_argument("--num-masks", type=int, help="Number of mask coefficients/prototypes (auto-inferred if omitted)")
    parser.add_argument("--class-names", nargs="+", help="Class names in model order")
    parser.add_argument("--normalize", action="store_true", help="Normalize input to [0,1]")
    parser.add_argument("--output", type=str, help="Optional JSON output path")
    args = parser.parse_args()

    if bool(args.image) == bool(args.images_dir):
        parser.error("Provide exactly one of --image or --images-dir")
    if args.class_names and args.num_classes is not None and args.num_classes != len(args.class_names):
        parser.error("--num-classes must match the number of values passed to --class-names")
    benchmark(args)
