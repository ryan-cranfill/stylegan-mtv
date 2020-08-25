import os
from pathlib import Path

from src.settings import OUTPUT_DIR


def uniquify(path: [str, Path]):
    if isinstance(path, Path):
        path = str(path)

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def make_output_path(input_path: [str, Path], out_dir=OUTPUT_DIR):
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)

    out_path = out_dir / input_path.with_suffix('.mp4').parts[-1]
    return uniquify(out_path)
