from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

SAM2p1_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


def download_sam2(model_name: str, checkpoint_dir: str | Path) -> Path:
    """Downloads a specified SAM2 model checkpoint if it does not exist.

    Args:
        model_name (str): The filename of the SAM2 model checkpoint
            (e.g., 'sam2.1_hiera_large.pt').
        checkpoint_dir (str | Path): The local directory path where the
            checkpoint will be saved.

    Returns:
        Path: The Path object pointing to the local checkpoint file.

    Raises:
        ValueError: If the `model_name` is empty or invalid.
        RuntimeError: If directory creation fails or network issues occur.
    """
    if not model_name:
        raise ValueError("The 'model_name' argument cannot be empty.")

    try:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create directory structure at '{checkpoint_dir}': {e}"
        )

    destination_path = checkpoint_path / model_name

    if not destination_path.exists():
        url = f"{SAM2p1_BASE_URL}/{model_name}"
        try:
            urlretrieve(url, destination_path)
        except (URLError, Exception) as e:
            if destination_path.exists():
                destination_path.unlink()
            raise RuntimeError(
                f"Failed to download checkpoint '{model_name}' "
                f"from remote source: {e}"
            )

    return destination_path