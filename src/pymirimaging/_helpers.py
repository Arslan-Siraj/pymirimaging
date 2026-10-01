from pathlib import Path
import json
import numpy as np


def value_of(value):
    return value.value if hasattr(value, "value") else str(value)


def format_wavenumbers(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return ",".join(f"{float(v):.12g}" for v in values)


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def find_info_file(source, info_path=None):
    source = Path(source).resolve()

    if info_path is not None:
        info = Path(info_path).resolve()
        if not info.is_file():
            raise FileNotFoundError(info)
        return info

    for parent in source.parents:
        candidate = parent / f"{parent.name}_INFO.txt"
        if candidate.is_file():
            return candidate

    return None


def read_pixel_sizes(source, info_path=None):
    info = find_info_file(source, info_path)
    if info is None:
        raise FileNotFoundError(
            "Slide INFO.txt not found. Pass info_path explicitly or "
            "store geometry attributes in the Zarr group."
        )

    fields = {}
    for line in info.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().casefold()] = value.strip()

    try:
        px = float(fields["x pixelsize"])
        py = float(fields["y pixelsize"])
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Missing or invalid X/Y pixelsize in {info}"
        ) from exc

    if any(not np.isfinite(v) or v <= 0 for v in (px, py)):
        raise ValueError(
            f"Pixel sizes must be positive finite values: {info}"
        )

    return px, py, info
