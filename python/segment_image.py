"""Single image segmentation with mask visualization (Hybrid Hailo + Python)."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common import (
    HailoSegmentationInferenceEngine,
    SegmentationPostProcessor,
    infer_segmentation_config_from_hef,
    scale_detections_to_original,
    format_detection_results,
    print_detection_summary,
)


def _build_engine(args):
    inferred = infer_segmentation_config_from_hef(args.hef)
    resolved_num_classes = args.num_classes or inferred["num_classes"]
    resolved_num_masks = args.num_masks or inferred["num_masks"]
    resolved_class_names = args.class_names or inferred["class_names"]

    if resolved_num_classes is None:
        raise ValueError("Unable to infer --num-classes from the HEF. Please pass it explicitly.")
    if resolved_num_masks is None:
        raise ValueError("Unable to infer --num-masks from the HEF. Please pass it explicitly.")
    if resolved_class_names is None:
        resolved_class_names = [f"class_{i}" for i in range(resolved_num_classes)]
    if len(resolved_class_names) != resolved_num_classes:
        raise ValueError(
            f"Class names mismatch: got {len(resolved_class_names)} names for {resolved_num_classes} classes"
        )

    return HailoSegmentationInferenceEngine(
        hef_path=args.hef,
        num_classes=resolved_num_classes,
        num_masks=resolved_num_masks,
        class_names=resolved_class_names,
    )


def _draw_boxes_only(image, detections):
    out = image.copy()
    for det in detections:
        x1 = int(max(0, det["x1"]))
        y1 = int(max(0, det["y1"]))
        x2 = int(max(0, det["x2"]))
        y2 = int(max(0, det["y2"]))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.get('cls_name', 'obj')} {det.get('conf', 0):.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def segment_and_visualize(args):
    engine = None
    try:
        engine = _build_engine(args)
        orig_image = HailoSegmentationInferenceEngine.load_image(args.image)
        orig_h, orig_w = orig_image.shape[:2]
        input_data, _, scale, pad_w, pad_h = HailoSegmentationInferenceEngine.preprocess(
            args.image, normalize=args.normalize
        )

        t_start = time.perf_counter()
        detections, masks, stats = engine.infer(
            input_data,
            verbose=args.verbose,
            conf_threshold=args.conf_threshold,
            mask_threshold=args.mask_threshold,
            iou_threshold=args.iou_threshold,
        )

        total_time = time.perf_counter() - t_start
        print(f"Inference completed in {total_time*1000:.2f}ms")
        print(f"  - Hailo: {stats.hailo_inference_time*1000:.2f}ms")
        print(f"  - Python Head: {stats.postprocess_time*1000:.2f}ms")
        print(format_detection_results(detections))

        detections = scale_detections_to_original(detections, orig_h, orig_w, scale, pad_w, pad_h)

        if masks is not None:
            output_image = SegmentationPostProcessor.draw_masks(
                orig_image, detections, masks, scale=scale, pad_w=pad_w, pad_h=pad_h, alpha=0.4
            )
        else:
            output_image = _draw_boxes_only(orig_image, detections)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_image)

        print_detection_summary(
            title="SEGMENTATION SUMMARY (Hybrid Hailo + Python Head)",
            image_path=args.image,
            model_info={"HEF": args.hef},
            total_time_ms=total_time * 1000,
            conf_threshold=args.conf_threshold,
            num_detections=len(detections),
            output_path=str(output_path),
        )
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Image Segmentation with Hailo-8L + Python Head")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--hef", type=str, required=True, help="Path to HEF model")
    parser.add_argument("--output", type=str, default="output_segmented.jpg", help="Output image path")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask binarization threshold")
    parser.add_argument("--num-classes", type=int, help="Number of classes (auto-inferred if omitted)")
    parser.add_argument("--num-masks", type=int, help="Number of mask prototypes (auto-inferred if omitted)")
    parser.add_argument("--class-names", nargs="+", help="Class names in order (auto-filled for COCO-80 HEFs)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--normalize", action="store_true", help="Normalize input to [0,1]")
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise SystemExit(f"Error: Image not found: {args.image}")
    if args.class_names and args.num_classes is not None and len(args.class_names) != args.num_classes:
        raise SystemExit("Error: --num-classes must match the number of values passed to --class-names")

    segment_and_visualize(args)
