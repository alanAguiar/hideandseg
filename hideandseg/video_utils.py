import gc
import os
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def extract_frames(video_path: str, output_path: str) -> List[str]:
    """Decomposes a video into individual JPEG frames and saves them to disk.

    Args:
        video_path (str): Path to the source video file.
        output_path (str): Root directory where the 'frames' folder is built.

    Returns:
        List[str]: Sorted list of frame base names without extensions.

    Raises:
        FileNotFoundError: If the source video file does not exist.
        RuntimeError: If the video stream cannot be opened or read.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Source video file not found at: '{video_path}'"
        )

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise RuntimeError(
            f"Failed to open video stream for file: '{video_path}'"
        )

    frames_dir = os.path.join(output_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    count = 0
    try:
        while True:
            success, image = vidcap.read()
            if not success:
                break

            frame_name = f"{count:05d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, frame_name), image)
            count += 1
    finally:
        vidcap.release()

    if count == 0:
        raise RuntimeError(
            f"No frames could be extracted from video file: '{video_path}'"
        )

    frames = sorted(
        [
            os.path.splitext(x)[0]
            for x in os.listdir(frames_dir)
            if x.endswith(".jpg")
        ]
    )

    return frames


def save_masks(frame: str, logits: Any, output_path: str) -> None:
    """Processes model logits into binary masks and saves alpha-masked targets.

    Args:
        frame (str): The base name of the frame being processed.
        logits (Any): PyTorch Tensor or NumPy array containing raw model logits.
        output_path (str): Root directory for output assets.

    Raises:
        FileNotFoundError: If the original source frame cannot be found.
    """
    masks_dir = os.path.join(output_path, "masks")
    masked_dir = os.path.join(output_path, "masked_frames")
    frames_dir = os.path.join(output_path, "frames")

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)

    mask = (logits[0] > 0.0).cpu().numpy()[0]
    mask_array = mask.astype(np.uint8) * 255

    cv2.imwrite(os.path.join(masks_dir, f"{frame}.bmp"), mask_array)

    source_frame_path = os.path.join(frames_dir, f"{frame}.jpg")
    if not os.path.exists(source_frame_path):
        raise FileNotFoundError(
            f"Source frame for masking not found at: '{source_frame_path}'"
        )

    image_array = np.array(Image.open(source_frame_path))

    mask_merged = cv2.merge((mask_array, mask_array, mask_array))
    masked_image = cv2.bitwise_and(image_array, mask_merged)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2RGBA)
    masked_image[:, :, 3] = mask_array

    contours, _ = cv2.findContours(
        mask_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return

    try:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        if w > 0 and h > 0:
            masked_image = masked_image[y : y + h, x : x + w]
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(
                os.path.join(masked_dir, f"{frame}.png"), masked_image
            )
    except ValueError:
        pass


def get_simulated_heatmap_color(value: float, invert: bool = False) -> tuple:
    """Simulates a smooth Red-Yellow-Green color mapping directly via BGR.

    Args:
        value (float): A value between 0.0 and 1.0.
        invert (bool, optional): Whether to invert the mapping. Defaults to False.

    Returns:
        tuple: A BGR color tuple.
    """
    val = 1.0 - value if invert else value
    val = max(0.0, min(1.0, val))
    if val < 0.5:
        r = 255
        g = int(2 * val * 255)
    else:
        r = int(2 * (1.0 - val) * 255)
        g = 255
    return (0, g, r)  # BGR format


def compile_video(framerate: int, output_path: str) -> None:
    """Generates a segmented video overlaid with a scientific dashboard panel.

    Args:
        framerate (int): The desired frames per second for the output video.
        output_path (str): The root tracking and workspace directory path.
    """
    print(f"[INFO] Initializing video compilation at {framerate} FPS...")

    font = cv2.FONT_HERSHEY_SIMPLEX
    color_bg = (20, 20, 20)  # Dark graphite panel background
    color_text = (245, 245, 245)  # Clean crisp white
    color_subtext = (160, 160, 160)  # Muted gray for static labels
    color_border = (50, 50, 50)  # Subtle dashboard split lines
    color_mask_edge = (255, 0, 255)  # Magenta boundaries for contours

    csv_path = os.path.join(output_path, "segmentation_quality_metrics.csv")
    df_metrics = pd.read_csv(csv_path)

    # Retain original NaN representation to handle transparency checks cleanly
    df_metrics["is_nan"] = df_metrics["dice"].isna()

    # Handle cases where DICE is missing/NaN by flagging errors
    df_metrics["nc_cleaned"] = np.where(
        df_metrics["is_nan"], np.inf, df_metrics["nc"]
    )

    # Normalized color value representations for continuous timelines
    df_metrics["nc_norm"] = (df_metrics["nc_cleaned"] / 10.0).clip(0.0, 1.0)
    df_metrics["dice_norm"] = (
        (df_metrics["dice"] - 0.75) / (1.0 - 0.75)
    ).clip(0.0, 1.0)

    # Get frame collection list
    frame_files = sorted(os.listdir(os.path.join(output_path, "frames")))
    if len(frame_files) > len(df_metrics):
        frame_files = [
            f
            for f in frame_files
            if not f.endswith("00000.jpg") and not f.endswith("00000.png")
        ]

    # Dynamically parse the actual dimensions of the raw files
    sample_img = cv2.imread(
        os.path.join(output_path, "frames", frame_files[0])
    )
    h_orig, w_orig, _ = sample_img.shape
    del sample_img

    h_panel = 180
    total_height = h_orig + h_panel
    total_width = w_orig

    # Native MP4v container deployment
    output_file = os.path.join(output_path, "segmented_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_file, fourcc, framerate, (total_width, total_height)
    )

    # Wrap the frame_files list in tqdm to display a progress bar
    for idx, frame_name in enumerate(
        tqdm(frame_files, desc="[INFO] Rendering Frames", unit="frame")
    ):
        frame_idx = idx + 1

        # 1. Load unmodified native components
        img_bgr = cv2.imread(os.path.join(output_path, "frames", frame_name))

        mask_file = os.path.join(
            output_path, "masks", f"{frame_name.split('.')[0]}.bmp"
        )
        if os.path.exists(mask_file):
            mask_gray = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        else:
            mask_gray = np.zeros((h_orig, w_orig), dtype=np.uint8)

        # 2. Build upper canvas: High-contrast full-frame overlay
        canvas_background = (img_bgr * 0.35).astype(np.uint8)
        canvas_superior = np.where(
            mask_gray[:, :, None] > 0, img_bgr, canvas_background
        )

        contours, _ = cv2.findContours(
            mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            canvas_superior, contours, -1, color_mask_edge, 2, cv2.LINE_AA
        )

        # 3. Extract dynamic metrics
        current_metrics = df_metrics[df_metrics.frame == frame_idx]
        val_nc = (
            current_metrics["nc_cleaned"].iloc[0]
            if not current_metrics.empty
            else np.inf
        )
        val_dice = (
            current_metrics["dice"].iloc[0]
            if not current_metrics.empty
            else np.nan
        )

        # 4. Build Lower Canvas: Scientific Dark Dashboard Panel
        dashboard = np.ones((h_panel, w_orig, 3), dtype=np.uint8) * color_bg[0]

        timeline_x_start = 280
        timeline_width = w_orig - timeline_x_start - 60
        pixel_per_frame = timeline_width / len(df_metrics)

        # Render NC (Number of Labels) Timeline
        y_nc = 45
        nc_text = f"NC(t) = {int(val_nc) if val_nc != np.inf else 'NaN'}"
        cv2.putText(
            dashboard,
            nc_text,
            (30, y_nc + 14),
            font,
            0.55,
            color_text,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            dashboard,
            "SEGMENTATION QUALITY",
            (timeline_x_start, y_nc - 12),
            font,
            0.45,
            color_subtext,
            1,
            cv2.LINE_AA,
        )

        # Render DICE Coefficient Timeline
        y_dice = 115
        dice_text = (
            f"DICE(t) = {val_dice:.3f}"
            if not np.isnan(val_dice)
            else "DICE(t) = NaN"
        )
        cv2.putText(
            dashboard,
            dice_text,
            (30, y_dice + 14),
            font,
            0.55,
            color_text,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            dashboard,
            "SEGMENTATION CONSISTENCY",
            (timeline_x_start, y_dice - 12),
            font,
            0.45,
            color_subtext,
            1,
            cv2.LINE_AA,
        )

        # Draw continuous color-mapped status histories
        for f_idx in range(len(df_metrics)):
            row = df_metrics.iloc[f_idx]
            x_f_start = int(timeline_x_start + (f_idx * pixel_per_frame))
            x_f_end = int(timeline_x_start + ((f_idx + 1) * pixel_per_frame))

            if row["is_nan"]:
                color_nc = color_bg
                color_dice = color_bg
            else:
                color_nc = get_simulated_heatmap_color(
                    row["nc_norm"], invert=True
                )
                color_dice = get_simulated_heatmap_color(
                    row["dice_norm"], invert=False
                )

            cv2.rectangle(
                dashboard, (x_f_start, y_nc), (x_f_end, y_nc + 18), color_nc, -1
            )
            cv2.rectangle(
                dashboard,
                (x_f_start, y_dice),
                (x_f_end, y_dice + 18),
                color_dice,
                -1,
            )

        # Render dynamic playhead indicators
        current_playhead_x = int(timeline_x_start + (idx * pixel_per_frame))
        cv2.line(
            dashboard,
            (current_playhead_x, y_nc - 5),
            (current_playhead_x, y_dice + 23),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            dashboard,
            (current_playhead_x, y_nc - 5),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            dashboard,
            (current_playhead_x, y_dice + 23),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        # Metadata string
        meta_text = (
            f"FRAME: {frame_idx:04d} / {len(df_metrics):04d}   |   "
            f"RATE: {framerate} FPS"
        )
        cv2.putText(
            dashboard,
            meta_text,
            (timeline_x_start, h_panel - 15),
            font,
            0.45,
            color_subtext,
            1,
            cv2.LINE_AA,
        )

        # 5. Composite and push straight to the video stream object
        cv2.line(dashboard, (0, 0), (w_orig, 0), color_border, 1)
        frame_final = np.vstack((canvas_superior, dashboard))

        # Write directly to container
        video_writer.write(frame_final)

    # Close context managers gracefully
    video_writer.release()
    gc.collect()

    print(f"\n[SUCCESS] Compilation complete! Video saved to: {output_file}")