"""Common utilities for Hailo-8L inference on YOLO26 (NMS-free dual-head model)"""

import numpy as np
import cv2
import time
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Mapping, Sequence

try:
    from hailo_platform import (
        VDevice,
        HEF,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        InputVStreams,
        OutputVStreams,
        InferVStreams,
        HailoStreamInterface,
        FormatType,
    )
    HAILO_PLATFORM_IMPORT_ERROR = None
except ImportError as exc:
    VDevice = None
    HEF = None
    ConfigureParams = None
    InputVStreamParams = None
    OutputVStreamParams = None
    InputVStreams = None
    OutputVStreams = None
    InferVStreams = None
    HailoStreamInterface = None
    FormatType = None
    HAILO_PLATFORM_IMPORT_ERROR = exc


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Vectorized sigmoid"""
    return 1.0 / (1.0 + np.exp(-x))


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Simple NMS over [x1, y1, x2, y2] boxes."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def _require_hailo_platform() -> None:
    if HAILO_PLATFORM_IMPORT_ERROR is not None:
        raise ImportError(
            "hailo_platform could not be imported. "
            "Verify that the active environment matches the installed HailoRT shared library. "
            f"Original error: {HAILO_PLATFORM_IMPORT_ERROR}"
        )


# ============================================================================
# Common Image Operations
# ============================================================================

def letterbox_image(img, target_size=640, color=(114, 114, 114)):
    """Resize image with aspect ratio preservation (letterbox)"""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return padded, scale, pad_w, pad_h


def load_and_preprocess_image(
    img_path: str, target_size: int = 640, normalize: bool = False
) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
    """Load and preprocess image for inference

    Args:
        img_path: Path to input image
        target_size: Target size for inference (default 640x640)
        normalize: If True, normalize to [0,1]; if False, keep as uint8 [0,255]

    Returns:
        (input_tensor, original_size, scale, pad_w, pad_h)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # YOLO models expect RGB, but cv2.imread loads as BGR, so convert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_rgb_image(img, target_size=target_size, normalize=normalize)


def preprocess_rgb_image(
    image_rgb: np.ndarray, target_size: int = 640, normalize: bool = False
) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
    """Preprocess an RGB image array using the same letterbox path as runtime."""
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Input image is empty")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_rgb.shape}")

    orig_h, orig_w = image_rgb.shape[:2]

    padded, scale, pad_w, pad_h = letterbox_image(image_rgb, target_size)

    if normalize:
        padded = padded.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(padded, axis=0)
    else:
        input_tensor = np.expand_dims(padded, axis=0).astype(np.uint8)

    return input_tensor, (orig_h, orig_w), scale, pad_w, pad_h


def scale_detections_to_original(
    detections: List[dict], orig_h: int, orig_w: int, scale: float, pad_w: int, pad_h: int
) -> List[dict]:
    """Scale detection coordinates from inference space (640x640) to original image space"""
    for det in detections:
        det["x1"] = max(0, min((det["x1"] - pad_w) / scale, orig_w))
        det["y1"] = max(0, min((det["y1"] - pad_h) / scale, orig_h))
        det["x2"] = max(0, min((det["x2"] - pad_w) / scale, orig_w))
        det["y2"] = max(0, min((det["y2"] - pad_h) / scale, orig_h))

    return detections


def format_detection_results(detections: List[dict], show_count: int = 10) -> str:
    """Format detection results for display"""
    lines = []
    for i, det in enumerate(detections[:show_count] if show_count else detections):
        lines.append(
            f"  [{i+1}] {det['cls_name']} - conf={det['conf']:.2f}, "
            f"bbox=[{det['x1']:.0f}, {det['y1']:.0f}, {det['x2']:.0f}, {det['y2']:.0f}]"
        )
    return "\n".join(lines)


def print_detection_summary(
    title: str,
    image_path: str,
    model_info: Dict,
    total_time_ms: float,
    conf_threshold: float,
    num_detections: int,
    output_path: str,
):
    """Print formatted detection summary"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Image: {image_path}")
    for key, val in model_info.items():
        print(f"Model {key}: {val}")
    print(f"Total Time: {total_time_ms:.2f}ms")
    print(f"Confidence Threshold: {conf_threshold}")
    print(f"Detections: {num_detections}")
    print(f"Output: {output_path}")


def get_coco_class_names() -> List[str]:
    """Return COCO class names in index order."""
    classes = DetectionPostProcessor._load_coco_classes()
    return [classes[i] for i in sorted(classes.keys())]


def default_class_names(num_classes: int) -> List[str]:
    """Return sensible default class names for a segmentation model."""
    if num_classes == 80:
        return get_coco_class_names()
    return [f"class_{i}" for i in range(num_classes)]


def resolve_class_names(
    num_classes: int, class_names: Optional[Sequence[str] | Mapping[int, str]] = None
) -> Dict[int, str]:
    """Return a stable class-id to class-name mapping."""
    if class_names is None:
        names = default_class_names(num_classes)
        return {idx: name for idx, name in enumerate(names)}

    if isinstance(class_names, Mapping):
        resolved = {int(idx): str(name) for idx, name in class_names.items()}
    else:
        resolved = {idx: str(name) for idx, name in enumerate(class_names)}

    missing = [idx for idx in range(num_classes) if idx not in resolved]
    if missing:
        raise ValueError(f"Missing class names for indices: {missing}")
    return resolved


def infer_segmentation_config_from_hef(hef_path: str) -> Dict[str, Optional[object]]:
    """Infer segmentation metadata by parsing the HEF with hailortcli."""
    cmd = ["hailortcli", "parse-hef", hef_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to parse HEF with hailortcli: {hef_path}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    output_shapes = []
    network_group_name = None

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Network group name:"):
            network_group_name = stripped.split(":", 1)[1].split(",", 1)[0].strip()

        if "Output " not in stripped:
            continue

        match = re.search(r"\((\d+)x(\d+)x(\d+)\)", stripped)
        if not match:
            continue

        h, w, c = map(int, match.groups())
        output_shapes.append((h, w, c, stripped))

    proto_candidates = [shape for shape in output_shapes if shape[0] == 160 and shape[1] == 160]
    num_masks = proto_candidates[0][2] if proto_candidates else None

    cls_candidates = []
    for h, w, c, raw in output_shapes:
        if (h, w) not in {(80, 80), (40, 40), (20, 20)}:
            continue
        if c == 4 or c == num_masks:
            continue
        cls_candidates.append((h, w, c, raw))

    num_classes = cls_candidates[0][2] if cls_candidates else None
    class_names = default_class_names(num_classes) if num_classes is not None else None

    return {
        "network_group_name": network_group_name,
        "num_classes": num_classes,
        "num_masks": num_masks,
        "class_names": class_names,
        "parse_output": result.stdout,
    }


@dataclass
class InferenceStats:
    preprocess_time: float
    hailo_inference_time: float
    postprocess_time: float
    total_time: float
    hailo_output_shape: str
    final_output_shape: str


class HailoPythonInferenceEngine:
    """Encapsulates Hailo-8L backbone + Python head inference"""

    def __init__(self, hef_path: str):
        _require_hailo_platform()
        self.hef_path = hef_path
        self.target = VDevice()
        self.hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )
        self.input_stream_name = self.hef.get_input_vstream_infos()[0].name
        self.shape_to_name = {
            (1, 80, 80, 80): "cls_80",
            (1, 40, 40, 80): "cls_40",
            (1, 20, 20, 80): "cls_20",
            (1, 80, 80, 4): "reg_80",
            (1, 40, 40, 4): "reg_40",
            (1, 20, 20, 4): "reg_20",
        }
        print(f"✓ Hailo engine initialized: {hef_path}")
        print("✓ Using Python head for post-processing.")

    @staticmethod
    def preprocess(
        img_path: str, width: int = 640, height: int = 640, normalize: bool = False
    ) -> Tuple[np.ndarray, Tuple[int, int], float, int, int]:
        return load_and_preprocess_image(img_path, width, normalize)

    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img

    def _run_python_head(self, dequantized_results: Dict, conf_threshold: float) -> List[dict]:
        tensors = {}
        found_shapes = []
        for _, data in dequantized_results.items():
            shape = data.shape
            found_shapes.append(shape)
            if shape in self.shape_to_name:
                tensors[self.shape_to_name[shape]] = data

        required_tensors = ["cls_80", "cls_40", "cls_20", "reg_80", "reg_40", "reg_20"]
        missing = [t for t in required_tensors if t not in tensors]
        if missing:
            print(f"Error: Missing tensors from HEF: {missing}")
            print(f"Found shapes: {found_shapes}")
            return []

        strides = [8, 16, 32]
        grid_sizes = [80, 40, 20]
        logit_threshold = -np.log(1.0 / conf_threshold - 1.0)
        results = []
        coco_classes = DetectionPostProcessor._load_coco_classes()

        for scale_idx in range(len(strides)):
            stride = strides[scale_idx]
            grid_dim = grid_sizes[scale_idx]
            cls_data = tensors[f"cls_{grid_dim}"][0]
            reg_data = tensors[f"reg_{grid_dim}"][0]
            cls_flat = cls_data.reshape(-1, 80)
            reg_flat = reg_data.reshape(-1, 4)
            max_logits = cls_flat.max(axis=1)
            class_ids = cls_flat.argmax(axis=1)
            mask = max_logits > logit_threshold
            if not mask.any():
                continue
            indices = np.where(mask)[0]
            scores = sigmoid(max_logits[indices])
            cls = class_ids[indices]
            rows = indices // grid_dim
            cols = indices % grid_dim
            l = reg_flat[indices, 0]
            t = reg_flat[indices, 1]
            r = reg_flat[indices, 2]
            b = reg_flat[indices, 3]
            x1 = (cols + 0.5 - l) * stride
            y1 = (rows + 0.5 - t) * stride
            x2 = (cols + 0.5 + r) * stride
            y2 = (rows + 0.5 + b) * stride
            for j in range(len(indices)):
                results.append(
                    {
                        "x1": round(float(x1[j]), 2),
                        "y1": round(float(y1[j]), 2),
                        "x2": round(float(x2[j]), 2),
                        "y2": round(float(y2[j]), 2),
                        "conf": round(float(scores[j]), 4),
                        "cls_id": int(cls[j]),
                        "cls_name": coco_classes.get(int(cls[j]), "N/A"),
                    }
                )

        return results

    def infer(
        self,
        input_data: np.ndarray,
        verbose: bool = False,
        save_output: bool = False,
        conf_threshold: float = 0.5,
    ) -> Tuple[List[dict], InferenceStats]:
        _ = save_output
        stats = InferenceStats(0, 0, 0, 0, "", "")
        t_start = time.perf_counter()
        if verbose:
            print(f"[INFERENCE] Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        with self.network_group.activate():
            with InferVStreams(
                self.network_group, self.input_vstream_params, self.output_vstream_params
            ) as infer_pipeline:
                t_hailo = time.perf_counter()
                hailo_results = infer_pipeline.infer({self.input_stream_name: input_data})
                stats.hailo_inference_time = time.perf_counter() - t_hailo
                stats.hailo_output_shape = str({k: v.shape for k, v in hailo_results.items()})
                t_post = time.perf_counter()
                detections = self._run_python_head(hailo_results, conf_threshold)
                stats.postprocess_time = time.perf_counter() - t_post
                stats.final_output_shape = f"{len(detections)} detections"
        stats.total_time = time.perf_counter() - t_start
        return detections, stats

    def close(self):
        if hasattr(self, "target") and self.target is not None:
            self.target.release()
            self.target = None


class DetectionPostProcessor:
    COCO_CLASSES = None

    @classmethod
    def _load_coco_classes(cls):
        if cls.COCO_CLASSES is not None:
            return cls.COCO_CLASSES
        cls.COCO_CLASSES = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
            6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
            11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
            16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
            22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
            27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
            32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
            36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
            40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
            45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
            50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
            55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
            60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
            65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
            70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
            75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
        }
        return cls.COCO_CLASSES

    @staticmethod
    def postprocess(detections: np.ndarray, conf_threshold: float = 0.5) -> List[dict]:
        classes = DetectionPostProcessor._load_coco_classes()
        results = []
        for det in detections:
            conf = det[4]
            if conf >= conf_threshold:
                cls_id = int(det[5])
                cls_name = classes.get(cls_id) if isinstance(classes, dict) else classes[cls_id]
                results.append(
                    {
                        "x1": float(det[0]),
                        "y1": float(det[1]),
                        "x2": float(det[2]),
                        "y2": float(det[3]),
                        "conf": float(conf),
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                    }
                )
        return results

    @staticmethod
    def draw_bboxes(image: np.ndarray, detections: list, thickness: int = 2) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, det in enumerate(detections):
            x1 = int(max(0, det["x1"]))
            y1 = int(max(0, det["y1"]))
            x2 = int(min(w, det["x2"]))
            y2 = int(min(h, det["y2"]))
            color = colors[i % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"{det['cls_name']} {det['conf']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img


# ============================================================================
# Segmentation Support
# ============================================================================

class HailoSegmentationInferenceEngine:
    """Hailo-8L backbone + Python head for YOLO26-seg instance segmentation."""

    def __init__(
        self,
        hef_path: str,
        num_classes: int = 80,
        num_masks: int = 32,
        class_names: Optional[Sequence[str] | Mapping[int, str]] = None,
    ):
        _require_hailo_platform()
        self.hef_path = hef_path
        self.num_classes = num_classes
        self.num_masks = num_masks
        self.class_names = resolve_class_names(num_classes, class_names)

        self.target = VDevice()
        self.hef = HEF(hef_path)
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        self.input_name = self.hef.get_input_vstream_infos()[0].name
        self.input_shape = tuple(self.hef.get_input_vstream_infos()[0].shape)

        nc = self.num_classes
        nm = self.num_masks
        self.shape_to_name = {
            (80, 80, nc): "cls_80",
            (40, 40, nc): "cls_40",
            (20, 20, nc): "cls_20",
            (80, 80, 4): "reg_80",
            (40, 40, 4): "reg_40",
            (20, 20, 4): "reg_20",
            (80, 80, nm): "mc_80",
            (40, 40, nm): "mc_40",
            (20, 20, nm): "mc_20",
            (160, 160, nm): "proto",
            (1, 80, 80, nc): "cls_80",
            (1, 40, 40, nc): "cls_40",
            (1, 20, 20, nc): "cls_20",
            (1, 80, 80, 4): "reg_80",
            (1, 40, 40, 4): "reg_40",
            (1, 20, 20, 4): "reg_20",
            (1, 80, 80, nm): "mc_80",
            (1, 40, 40, nm): "mc_40",
            (1, 20, 20, nm): "mc_20",
            (1, 160, 160, nm): "proto",
        }

        self._ng_ctx = self.network_group.activate()
        self._ng_ctx.__enter__()
        self._ivs_ctx = InputVStreams(self.network_group, self.input_vstream_params)
        self._input_vstreams = self._ivs_ctx.__enter__()
        self._ovs_ctx = OutputVStreams(self.network_group, self.output_vstream_params)
        self._output_vstreams = self._ovs_ctx.__enter__()

    @staticmethod
    def preprocess(img_path: str, width: int = 640, height: int = 640, normalize: bool = False):
        _ = height
        return load_and_preprocess_image(img_path, width, normalize)

    @staticmethod
    def load_image(img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return img

    def _run_python_head(
        self, dequantized_results: Dict, conf_threshold: float, iou_threshold: float = 0.5
    ) -> Tuple[List[dict], np.ndarray]:
        nc = self.num_classes
        nm = self.num_masks
        tensors = {}
        found_shapes = []
        for _, data in dequantized_results.items():
            shape = data.shape
            found_shapes.append(shape)
            if shape in self.shape_to_name:
                tensors[self.shape_to_name[shape]] = data

        required = ["cls_80", "cls_40", "cls_20", "reg_80", "reg_40", "reg_20", "mc_80", "mc_40", "mc_20"]
        missing = [t for t in required if t not in tensors]
        if missing:
            print(f"Error: Missing tensors: {missing}")
            print(f"Found shapes: {found_shapes}")
            return [], None

        if "proto" in tensors:
            if tensors["proto"].ndim == 4:
                proto = tensors["proto"][0].transpose(2, 0, 1)
            elif tensors["proto"].ndim == 3:
                proto = tensors["proto"].transpose(2, 0, 1)
            else:
                return [], None
        else:
            return [], None

        strides = [8, 16, 32]
        grid_sizes = [80, 40, 20]
        logit_threshold = -np.log(1.0 / conf_threshold - 1.0)
        all_boxes, all_scores, all_cls, all_coeffs = [], [], [], []

        for scale_idx in range(len(strides)):
            stride = strides[scale_idx]
            grid_dim = grid_sizes[scale_idx]
            cls_data = tensors[f"cls_{grid_dim}"][0] if tensors[f"cls_{grid_dim}"].ndim == 4 else tensors[f"cls_{grid_dim}"]
            reg_data = tensors[f"reg_{grid_dim}"][0] if tensors[f"reg_{grid_dim}"].ndim == 4 else tensors[f"reg_{grid_dim}"]
            mc_data = tensors[f"mc_{grid_dim}"][0] if tensors[f"mc_{grid_dim}"].ndim == 4 else tensors[f"mc_{grid_dim}"]
            cls_flat = cls_data.reshape(-1, nc)
            reg_flat = reg_data.reshape(-1, 4)
            mc_flat = mc_data.reshape(-1, nm)
            max_logits = cls_flat.max(axis=1)
            class_ids = cls_flat.argmax(axis=1)
            mask = max_logits > logit_threshold
            if not mask.any():
                continue
            indices = np.where(mask)[0]
            scores = sigmoid(max_logits[indices])
            cls = class_ids[indices]
            rows = indices // grid_dim
            cols = indices % grid_dim
            l = reg_flat[indices, 0]
            t = reg_flat[indices, 1]
            r = reg_flat[indices, 2]
            b = reg_flat[indices, 3]
            x1 = (cols + 0.5 - l) * stride
            y1 = (rows + 0.5 - t) * stride
            x2 = (cols + 0.5 + r) * stride
            y2 = (rows + 0.5 + b) * stride
            coeffs = mc_flat[indices]
            all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
            all_scores.append(scores)
            all_cls.append(cls)
            all_coeffs.append(coeffs)

        if not all_boxes:
            return [], proto

        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_cls = np.concatenate(all_cls, axis=0)
        all_coeffs = np.concatenate(all_coeffs, axis=0)
        keep = []
        for class_id in range(nc):
            class_mask = all_cls == class_id
            if not class_mask.any():
                continue
            class_indices = np.where(class_mask)[0]
            class_keep = nms_boxes(all_boxes[class_indices], all_scores[class_indices], iou_threshold)
            keep.extend(class_indices[class_keep].tolist())

        results = []
        for idx in keep:
            cls_id = int(all_cls[idx])
            results.append(
                {
                    "x1": round(float(all_boxes[idx, 0]), 2),
                    "y1": round(float(all_boxes[idx, 1]), 2),
                    "x2": round(float(all_boxes[idx, 2]), 2),
                    "y2": round(float(all_boxes[idx, 3]), 2),
                    "conf": round(float(all_scores[idx]), 4),
                    "cls_id": cls_id,
                    "cls_name": self.class_names.get(cls_id, "N/A"),
                    "mask_coeffs": all_coeffs[idx],
                }
            )
        return results, proto

    def infer(
        self,
        input_data: np.ndarray,
        verbose: bool = False,
        conf_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> Tuple[List[dict], np.ndarray, InferenceStats]:
        stats = InferenceStats(0, 0, 0, 0, "", "")
        t_start = time.perf_counter()
        if input_data.ndim != 4:
            raise ValueError(f"Input shape mismatch: got {input_data.shape}, expected (1, {self.input_shape})")
        frame_batch = np.ascontiguousarray(input_data, dtype=np.uint8)
        if tuple(frame_batch.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Input shape mismatch: got {tuple(frame_batch.shape[1:])}, expected {self.input_shape}"
            )
        if frame_batch.nbytes == 0:
            raise ValueError("Input buffer is empty")

        t_hailo = time.perf_counter()
        for ivs in self._input_vstreams:
            ivs.send(frame_batch)
            break
        hailo_results = {}
        for ovs in self._output_vstreams:
            hailo_results[ovs.info.name] = ovs.recv()
        stats.hailo_inference_time = time.perf_counter() - t_hailo

        t_post = time.perf_counter()
        detections, proto = self._run_python_head(hailo_results, conf_threshold, iou_threshold)
        instance_masks = None
        if detections and proto is not None:
            instance_masks = self._assemble_masks(detections, proto, mask_threshold)
        stats.postprocess_time = time.perf_counter() - t_post
        stats.total_time = time.perf_counter() - t_start
        stats.final_output_shape = f"{len(detections)} detections"
        return detections, instance_masks, stats

    def close(self):
        for streams_name in ("_output_vstreams", "_input_vstreams"):
            if hasattr(self, streams_name):
                setattr(self, streams_name, None)
        for ctx_name in ("_ovs_ctx", "_ivs_ctx", "_ng_ctx"):
            ctx = getattr(self, ctx_name, None)
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
                setattr(self, ctx_name, None)
        if hasattr(self, "network_group"):
            self.network_group = None
        if hasattr(self, "hef"):
            self.hef = None
        if hasattr(self, "target") and self.target is not None:
            self.target.release()
            self.target = None

    @staticmethod
    def _assemble_masks(
        detections: List[dict], proto: np.ndarray, mask_threshold: float = 0.5
    ) -> np.ndarray:
        nm, proto_h, proto_w = proto.shape
        coeffs = np.stack([d["mask_coeffs"] for d in detections], axis=0)
        masks_flat = coeffs @ proto.reshape(nm, -1)
        masks_flat = 1.0 / (1.0 + np.exp(-masks_flat))
        masks = masks_flat.reshape(len(detections), proto_h, proto_w)
        scale = proto_h / 640.0
        for i, det in enumerate(detections):
            x1 = max(0, int(det["x1"] * scale))
            y1 = max(0, int(det["y1"] * scale))
            x2 = min(proto_w, int(det["x2"] * scale) + 1)
            y2 = min(proto_h, int(det["y2"] * scale) + 1)
            crop_mask = np.zeros((proto_h, proto_w), dtype=np.float32)
            crop_mask[y1:y2, x1:x2] = masks[i, y1:y2, x1:x2]
            masks[i] = crop_mask
        return (masks > mask_threshold).astype(np.uint8)


class SegmentationPostProcessor:
    @staticmethod
    def draw_masks(
        image: np.ndarray,
        detections: List[dict],
        masks: np.ndarray,
        scale: float,
        pad_w: int,
        pad_h: int,
        alpha: float = 0.5,
        thickness: int = 2,
    ) -> np.ndarray:
        img = image.copy()
        orig_h, orig_w = img.shape[:2]
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 255, 0), (255, 128, 0), (0, 128, 255),
        ]
        if masks is None or len(detections) == 0:
            return img
        for i, det in enumerate(detections):
            color = colors[i % len(colors)]
            mask_160 = masks[i]
            mask_640 = cv2.resize(mask_160, (640, 640), interpolation=cv2.INTER_LINEAR)
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            mask_crop = mask_640[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
            mask_orig = cv2.resize(mask_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask_bool = mask_orig > 0.5
            overlay = img.copy()
            overlay[mask_bool] = color
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            x1 = int(max(0, det["x1"]))
            y1 = int(max(0, det["y1"]))
            x2 = int(min(orig_w, det["x2"]))
            y2 = int(min(orig_h, det["y2"]))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"{det['cls_name']} {det['conf']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img
