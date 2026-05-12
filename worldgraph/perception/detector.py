"""YOLOE + SAM2 perception pipeline.

Converts raw RGB frames into per-pixel semantic label images
suitable for Hydra's pipeline.step() input.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class PerceptionPipeline:
    """Open-vocabulary detection + segmentation pipeline.

    Pipeline: RGB → YOLOE (detect + segment) → per-pixel label image
    Optional: → SAM2 refinement for higher-quality masks
    """

    def __init__(
        self,
        yoloe_model: str = "yoloe-11l-seg.pt",
        yoloe_conf: float = 0.3,
        use_sam2: bool = False,
        sam2_model: str = "facebook/sam2-hiera-tiny",
        default_classes: Optional[List[str]] = None,
    ):
        self.yoloe_model_name = yoloe_model
        self.yoloe_conf = yoloe_conf
        self.use_sam2 = use_sam2
        self.sam2_model_name = sam2_model

        self._yoloe = None
        self._sam2_model = None
        self._sam2_processor = None

        # Default object classes for kitchen/home scenarios
        self.default_classes = default_classes or [
            "person", "chair", "table", "couch", "bed", "tv",
            "laptop", "phone", "bottle", "cup", "mug", "bowl",
            "plate", "fork", "knife", "spoon", "book", "clock",
            "vase", "scissors", "teddy bear", "toothbrush",
            "refrigerator", "oven", "microwave", "sink",
            "toaster", "banana", "apple", "orange", "sandwich",
            "broccoli", "carrot", "pizza", "cake", "donut",
            "keys", "wallet", "bag", "backpack", "umbrella",
            "remote", "keyboard", "mouse", "lamp", "plant",
        ]

    def _load_yoloe(self):
        if self._yoloe is not None:
            return
        from ultralytics import YOLO
        self._yoloe = YOLO(self.yoloe_model_name)

    def _load_sam2(self):
        if self._sam2_model is not None:
            return
        from transformers import Sam2Model, Sam2Processor
        self._sam2_processor = Sam2Processor.from_pretrained(self.sam2_model_name)
        self._sam2_model = Sam2Model.from_pretrained(self.sam2_model_name)
        self._sam2_model.eval()

    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[str]] = None,
    ) -> Dict:
        """Run YOLOE detection + segmentation.

        Args:
            image: [H, W, 3] uint8 RGB image.
            classes: Object classes to detect. Uses defaults if None.

        Returns:
            Dict with 'boxes', 'masks', 'labels', 'scores', 'label_ids'.
        """
        self._load_yoloe()
        classes = classes or self.default_classes

        # Set classes for open-vocab detection
        self._yoloe.set_classes(classes)

        # Run inference
        results = self._yoloe.predict(
            image,
            conf=self.yoloe_conf,
            verbose=False,
        )[0]

        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([])
        scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else np.array([])
        label_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else np.array([])
        labels = [classes[i] if i < len(classes) else f"class_{i}" for i in label_ids]

        # Extract masks if available (segmentation model)
        masks = None
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()  # [N, H, W]

        return {
            "boxes": boxes,
            "masks": masks,
            "labels": labels,
            "label_ids": label_ids,
            "scores": scores,
        }

    def build_label_image(
        self,
        image: np.ndarray,
        detections: Optional[Dict] = None,
        classes: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Build a per-pixel semantic label image for Hydra.

        Args:
            image: [H, W, 3] uint8 RGB image.
            detections: Pre-computed detections. If None, runs detect().
            classes: Object classes (used if detections is None).

        Returns:
            Tuple of:
            - label_image: [H, W] int32 array where each pixel = class ID
            - object_info: List of dicts with detection metadata
        """
        if detections is None:
            detections = self.detect(image, classes)

        H, W = image.shape[:2]
        label_image = np.zeros((H, W), dtype=np.int32)
        object_info = []

        masks = detections.get("masks")
        boxes = detections["boxes"]
        labels = detections["labels"]
        label_ids = detections["label_ids"]
        scores = detections["scores"]

        for i in range(len(boxes)):
            class_id = int(label_ids[i]) + 1  # +1 so 0 = background

            if masks is not None and i < len(masks):
                # Use segmentation mask
                mask = masks[i]
                if mask.shape != (H, W):
                    from PIL import Image as PILImage
                    mask_resized = np.array(
                        PILImage.fromarray(mask.astype(np.uint8)).resize((W, H))
                    )
                    mask = mask_resized > 0.5
                label_image[mask] = class_id
            else:
                # Fall back to bounding box fill
                x1, y1, x2, y2 = boxes[i].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                label_image[y1:y2, x1:x2] = class_id

            object_info.append({
                "class_id": class_id,
                "label": labels[i],
                "score": float(scores[i]),
                "bbox": boxes[i].tolist() if len(boxes) > i else [],
            })

        return label_image, object_info

    def extract_object_crops(
        self,
        image: np.ndarray,
        detections: Optional[Dict] = None,
        padding: int = 10,
    ) -> List[np.ndarray]:
        """Extract RGB crops for each detected object (for DINOv2).

        Args:
            image: [H, W, 3] uint8 RGB image.
            detections: Pre-computed detections.
            padding: Pixel padding around bounding boxes.

        Returns:
            List of [crop_H, crop_W, 3] uint8 arrays.
        """
        if detections is None:
            detections = self.detect(image)

        H, W = image.shape[:2]
        crops = []

        for box in detections["boxes"]:
            x1, y1, x2, y2 = box.astype(int)
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(W, x2 + padding)
            y2 = min(H, y2 + padding)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((32, 32, 3), dtype=np.uint8)
            crops.append(crop)

        return crops
