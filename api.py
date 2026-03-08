"""
FastAPI application exposing the mystery coloring generator as an HTTP API.
"""

import io
import os
import re
import zipfile
from typing import List, Optional, Tuple

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from config import Config
from main import MysteryColoringGenerator


app = FastAPI(
    title="Mystery Coloring API",
    description="Upload an image and receive the generated _combined and _preview files.",
    version="1.0.0",
)


def sanitize_output_name(filename: Optional[str]) -> str:
    """Create a safe base filename for generated assets."""
    raw_name = os.path.splitext(os.path.basename(filename or "image"))[0]
    cleaned_name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name).strip("_")
    return cleaned_name or "image"


def parse_forced_colors(value: Optional[str]) -> List[Tuple[int, int, int]]:
    """Parse forced RGB colors from the CLI-style semicolon format."""
    if not value:
        return []

    forced_colors: List[Tuple[int, int, int]] = []
    for color_str in value.split(";"):
        rgb = tuple(map(int, color_str.strip().split(",")))
        if len(rgb) != 3 or any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(f"Invalid RGB values: {color_str}")
        forced_colors.append((rgb[0], rgb[1], rgb[2]))

    return forced_colors


def build_config(
    colors: int,
    difficulty: int,
    symbols: str,
    min_area: int,
    resolution: int,
    symbol_size: float,
    prefill_dark: int,
    mode_filter: int,
    no_bilateral: bool,
    force_colors: Optional[str],
) -> Config:
    """Build a generator config from submitted form values."""
    if symbols not in {"numbers", "letters", "custom"}:
        raise ValueError("symbols must be one of: numbers, letters, custom")

    config = Config()
    config.output_dir = "output"
    config.legend_position = "bottom"
    config.num_colors = colors
    config.difficulty_level = difficulty
    config.symbol_type = symbols
    config.min_region_area = min_area
    config.max_image_dimension = resolution
    config.font_size_ratio = symbol_size
    config.prefill_dark_threshold = prefill_dark
    config.prefill_dark_regions = prefill_dark > 0
    config.mode_filter_size = mode_filter
    config.bilateral_filter_enabled = not no_bilateral
    config.forced_colors = parse_forced_colors(force_colors)
    config.validate()
    return config


def encode_png(image, label: str) -> bytes:
    """Encode a BGR image as PNG bytes."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Failed to encode {label} image")
    return buffer.tobytes()


@app.get("/")
def root() -> dict:
    """Basic info endpoint."""
    return {
        "name": "Mystery Coloring API",
        "docs": "/docs",
        "health": "/health",
        "generate": "/generate",
    }


@app.get("/health")
def health() -> dict:
    """Health endpoint for Render."""
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    output_name: Optional[str] = Form(None),
    colors: int = Form(16),
    difficulty: int = Form(5),
    symbols: str = Form("numbers"),
    min_area: int = Form(50),
    resolution: int = Form(1400),
    symbol_size: float = Form(0.5),
    prefill_dark: int = Form(500),
    mode_filter: int = Form(5),
    no_bilateral: bool = Form(False),
    force_colors: Optional[str] = Form(None),
):
    """
    Generate the mystery coloring assets from an uploaded image.

    Returns a ZIP containing `<name>_combined.png` and `<name>_preview.png`.
    """
    if not image.filename:
        raise HTTPException(status_code=400, detail="A filename is required for the uploaded image")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")

    base_name = sanitize_output_name(output_name or image.filename)

    try:
        config = build_config(
            colors=colors,
            difficulty=difficulty,
            symbols=symbols,
            min_area=min_area,
            resolution=resolution,
            symbol_size=symbol_size,
            prefill_dark=prefill_dark,
            mode_filter=mode_filter,
            no_bilateral=no_bilateral,
            force_colors=force_colors,
        )

        generator = MysteryColoringGenerator(config)
        loaded_image = generator.load_image_bytes(image_bytes, image.filename)
        outputs = generator.generate_images(loaded_image)

        combined_bytes = encode_png(outputs["combined"], "combined")
        preview_bytes = encode_png(outputs["preview"], "preview")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{base_name}_combined.png", combined_bytes)
        archive.writestr(f"{base_name}_preview.png", preview_bytes)

    archive_buffer.seek(0)

    headers = {
        "Content-Disposition": f'attachment; filename="{base_name}_outputs.zip"'
    }
    return StreamingResponse(archive_buffer, media_type="application/zip", headers=headers)
