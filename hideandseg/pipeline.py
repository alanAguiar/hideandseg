import os
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sam2.build_sam import build_sam2_video_predictor
from skimage.measure import label
import torch
from ultralytics import YOLO

from .download_checkpoint import download_sam2
from .video_utils import extract_frames, save_masks

BASE_PATH = Path(__file__).parent


class HideAndSeg:
    """Automated video object segmentation pipeline combining YOLO and SAM2.

    This class leverages YOLO for initial bounding box detection on selected
    reference frames and propagates those detections using SAM2 to generate
    continuous segmentation masks across the video timeline.

    Attributes:
        yolo (YOLO): The YOLO object detection model instance.
        device (torch.device): Hardware device used for model inference.
        sam2 (SAM2VideoPredictor): The SAM2 video predictor instance.
    """

    def __init__(
        self,
        sam2_model_size: str = "large",
        sam2_checkpoint_path: str = str(BASE_PATH / "models" / "sam2" / "checkpoints"),
        yolo_weights: str = str(BASE_PATH / "models" / "yolo" / "yolo_weights.pt"),
    ):
        """Initializes HideAndSeg with YOLO and SAM2 configurations.

        Args:
            sam2_model_size (str): Size of the SAM2 model ('tiny', 'large').
            sam2_checkpoint_path (str): Directory where checkpoints are stored.
            yolo_weights (str): Path to the YOLO weights file (.pt).

        Raises:
            FileNotFoundError: If the specified YOLO weights file does not exist.
            ValueError: If an invalid SAM2 model size is provided.
            RuntimeError: If initialization or checkpoint downloads fail.
        """
        print("[INFO] Initializing HideAndSeg pipeline...")

        if not os.path.exists(yolo_weights):
            raise FileNotFoundError(
                f"YOLO weights file not found at: '{yolo_weights}'"
            )

        try:
            self.yolo = YOLO(yolo_weights)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize YOLO model with weights "
                f"'{yolo_weights}': {e}"
            )

        self.device = torch.device(
            "cuda" if torch.torch.cuda.is_available() else "cpu"
        )
        print(f"[INFO] Running inference on device: {self.device}")

        sam2_checkpoint, sam2_config = self._load_sam2_config(sam2_model_size)

        try:
            print("[INFO] Verifying SAM2 checkpoint...")
            download_sam2(sam2_checkpoint, sam2_checkpoint_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download SAM2 checkpoint: {e}")

        full_config_path = os.path.join("configs/sam2.1", sam2_config)
        full_checkpoint_path = os.path.join(
            sam2_checkpoint_path, sam2_checkpoint
        )

        try:
            self.sam2 = build_sam2_video_predictor(
                full_config_path, full_checkpoint_path, device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build SAM2 video predictor: {e}")

        print("[SUCCESS] HideAndSeg pipeline initialized successfully.")

    def _load_sam2_config(self, model_size: str) -> Tuple[str, str]:
        """Maps the model size string to checkpoint and config filenames."""
        config_map = {
            "tiny": ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml"),
            "small": ("sam2.1_hiera_small.pt", "sam2.1_hiera_s.yaml"),
            "base": ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml"),
            "large": ("sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml"),
        }

        if model_size not in config_map:
            raise ValueError(
                f"Unknown SAM2 model size '{model_size}'. "
                f"Valid options are: {list(config_map.keys())}"
            )

        return config_map[model_size]

    def _get_yolo_bbox(self, frame_path: str) -> np.ndarray:
        """Runs YOLO inference on a single frame path and returns the bbox."""
        yolo_result = self.yolo(frame_path, verbose=False)

        if not yolo_result or len(yolo_result[0].boxes) == 0:
            raise RuntimeError(
                f"No objects detected by YOLO in frame: {frame_path}"
            )

        x, y, w, h = yolo_result[0].boxes.xywh[0].detach().cpu().numpy()

        box = np.array(
            [
                int(x - w / 2),
                int(y - h / 2),
                int(x + w / 2),
                int(y + h / 2),
            ],
            dtype=np.float32,
        )

        return box

    def _get_annotation_frames(
        self, frames: List[str], n_annotated_frames: int
    ) -> List[int]:
        """Calculates equidistantly spaced frame indices for reference."""
        n_annotated_frames = max(1, min(n_annotated_frames, len(frames)))
        selected_idxs = [0]

        total_frames = len(frames)
        frame_dist = (total_frames - 1) / (n_annotated_frames - 1)

        for i in range(1, n_annotated_frames):
            target_idx = frame_dist * i
            idx = min(
                range(total_frames), key=lambda x: abs(x - target_idx)
            )
            if idx not in selected_idxs:
                selected_idxs.append(idx)

        return sorted(selected_idxs)

    def _calculate_metrics(
        self, logits_0: torch.Tensor, logits_1: torch.Tensor
    ) -> Tuple[float, float]:
        """Computes unsupervised segmentation quality metrics."""
        mask_1 = (logits_1[0] > 0.0).cpu().numpy()[0]

        if logits_0 is None:
            if mask_1.sum() > 0:
                return np.nan, 1.0
            else:
                return np.nan, np.nan

        mask_0 = (logits_0[0] > 0.0).cpu().numpy()[0]

        # Calculate geometric changes
        diff_mask = mask_1 * (mask_0 != mask_1).astype(int)
        nc_val = np.unique((label(mask_1) + 1) * diff_mask).shape[0] - 1

        denominator = mask_0.sum() + mask_1.sum()
        if denominator > 0:
            dice_val = 2 * (mask_0 * mask_1).sum() / denominator
        else:
            dice_val = 1.0 if mask_1.sum() == 0 else 0.0

        return nc_val, dice_val

    def _print_quality_report(self, df: pd.DataFrame) -> None:
        """Compiles and renders a report of segmentation quality metrics."""
        total_records = len(df)
        if total_records == 0:
            return

        evaluated_df = df.dropna(subset=["dice"])
        valid_count = len(evaluated_df)

        mask_present = evaluated_df["dice"].notna().sum()
        pct_masks = (mask_present / total_records) * 100

        d_scores = evaluated_df["dice"]
        d_good = (d_scores >= 0.9).sum()
        d_med = ((d_scores >= 0.8) & (d_scores < 0.9)).sum()
        d_bad = (d_scores < 0.75).sum()

        nc_scores = evaluated_df["nc"]
        nc_good = (nc_scores <= 2).sum()
        nc_med = ((nc_scores >= 3) & (nc_scores <= 5)).sum()
        nc_bad = (nc_scores > 5).sum()

        anomalous = (
            evaluated_df[(d_scores < 0.75) | (nc_scores > 5)].index.tolist()
        )
        intervals = []
        if anomalous:
            start = prev = anomalous[0]
            for f in anomalous[1:] + [None]:
                if f == prev + 1:
                    prev = f
                else:
                    if start == prev:
                        intervals.append(f"Frame {start}")
                    else:
                        intervals.append(f"Frames {start}-{prev}")
                    if f is not None:
                        start = prev = f
        else:
            intervals.append(
                "None detected. Segmentation variance sits entirely "
                "within nominal thresholds."
            )

        border = "═" * 70
        sub_border = "─" * 70
        report = (
            f"\n{border}\n"
            f"                SEGMENTATION QUALITY EVALUATION REPORT                \n"
            f"{border}\n"
            f" Total Tracked Timeline Length  : {total_records} frames\n"
            f" Segmentation Coverage          : {pct_masks:.2f}% "
            f"({mask_present}/{total_records} frames)\n"
            f"{sub_border}\n"
            f" METRIC DISTRIBUTION ANALYSIS:\n"
            f"{sub_border}\n"
            f"  [DICE(t)]\n"
            f"    │  High Stability     (1.0 - 0.9) : "
            f"{(d_good / valid_count) * 100:6.2f}%  ({d_good} frames)\n"
            f"    │  Moderate Drift     (0.9 - 0.75): "
            f"{(d_med / valid_count) * 100:6.2f}%  ({d_med} frames)\n"
            f"    │  Stability Warning  (< 0.75)    : "
            f"{(d_bad / valid_count) * 100:6.2f}%  ({d_bad} frames)\n"
            f"  [NC(t)]\n"
            f"    │  Stable Geometry     (1 - 2)    : "
            f"{(nc_good / valid_count) * 100:6.2f}%  ({nc_good} frames)\n"
            f"    │  Minor Fragmentation (3 - 5)    : "
            f"{(nc_med / valid_count) * 100:6.2f}%  ({nc_med} frames)\n"
            f"    │  Disruption Warning  (5+)       : "
            f"{(nc_bad / valid_count) * 100:6.2f}%  ({nc_bad} frames)\n"
            f"{sub_border}\n"
            f" IDENTIFIED SECTIONS OF UNSTABILITY:\n"
            f"{sub_border}"
        )
        print(report)

        for interval in intervals[:6]:
            print(f"  » Unstable sections: {interval}")
        if len(intervals) > 6:
            print(
                f"  » ... and {len(intervals) - 6} more localized variations."
            )
        print(f"{border}\n")

    def segment(
        self,
        video_path: str,
        output_path: str,
        n_annotated_frames: int = 5,
    ) -> None:
        """Executes the complete video object segmentation workflow."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Input video file not found at: '{video_path}'"
            )

        os.makedirs(output_path, exist_ok=True)

        print("[INFO] Step 1/4: Extracting video frames...")
        frames = extract_frames(video_path, output_path)

        if not frames:
            raise RuntimeError(
                "Frame extraction failed or output directory is empty."
            )
        print(f"[SUCCESS] Successfully extracted {len(frames)} frames.")

        annotation_frames = self._get_annotation_frames(
            frames, n_annotated_frames
        )
        frames_dir = os.path.join(output_path, "frames")

        print("[INFO] Step 2/4: Initializing SAM2 engine state...")
        try:
            state = self.sam2.init_state(
                video_path=frames_dir, async_loading_frames=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Critical error initializing SAM2 video state: {e}"
            )

        print(
            f"[INFO] Step 3/4: Generating object annotations on "
            f"{len(annotation_frames)} keyframes..."
        )
        for ann in annotation_frames:
            frame_filename = (
                frames[ann] + ".jpg"
                if not frames[ann].endswith(".jpg")
                else frames[ann]
            )
            frame_full_path = os.path.join(frames_dir, frame_filename)

            try:
                box = self._get_yolo_bbox(frame_full_path)
                self.sam2.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=ann,
                    obj_id=1,
                    box=box,
                )
                print(f"  -> Keyframe {ann} annotated successfully.")
            except Exception as e:
                print(
                    f"  [WARNING] Skipping keyframe {ann} ({frame_filename}) "
                    f"due to error: {e}"
                )
                continue

            print(
                "[INFO] Step 4/4: Propagating segmentation masks "
                "throughout the video..."
            )

        previous_logits = None
        dice_list, nc_list = [], []

        try:
            for (
                out_frame_idx,
                _,
                out_mask_logits,
            ) in self.sam2.propagate_in_video(state):
                nc, dice = self._calculate_metrics(
                    previous_logits, out_mask_logits
                )
                dice_list.append(dice)
                nc_list.append(nc)
                previous_logits = out_mask_logits
                save_masks(frames[out_frame_idx], out_mask_logits, output_path)
            print("[SUCCESS] Video propagation completed.")
        except Exception as e:
            print(f"[ERROR] Error occurred during mask propagation: {e}")
        finally:
            self.sam2.reset_state(state)
            print("[INFO] SAM2 state cleaned up. Process finished.")

        metrics_df = pd.DataFrame(
            {"frame": frames, "dice": dice_list, "nc": nc_list}
        )

        self._print_quality_report(metrics_df)

        metrics_df.to_csv(
            os.path.join(output_path, "segmentation_quality_metrics.csv"),
            index=False,
        )
