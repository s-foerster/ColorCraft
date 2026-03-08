"""
FastAPI application exposing the mystery coloring generator as an HTTP API.
"""

import io
import os
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from supabase import Client, create_client

from config import Config
from main import MysteryColoringGenerator


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JOBS_TABLE = os.getenv("SUPABASE_JOBS_TABLE", "jobs")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "coloring-jobs")
JOB_WORKERS = int(os.getenv("JOB_WORKERS", "2"))
JOB_HEARTBEAT_SECONDS = int(os.getenv("JOB_HEARTBEAT_SECONDS", "15"))
JOB_STALE_SECONDS = int(os.getenv("JOB_STALE_SECONDS", "180"))

_SUPABASE_CLIENT: Optional[Client] = None
_ACTIVE_JOBS: set[str] = set()
_ACTIVE_JOBS_LOCK = threading.Lock()
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=JOB_WORKERS)

app = FastAPI(
    title="Mystery Coloring API",
    description="Upload an image, receive a job id immediately, then poll until outputs are ready.",
    version="3.0.0",
)


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def format_datetime(value: datetime) -> str:
    """Format datetimes consistently for API responses."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp from Supabase."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def get_supabase() -> Client:
    """Create the Supabase client lazily."""
    global _SUPABASE_CLIENT

    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is missing. Set SUPABASE_URL or VITE_SUPABASE_URL.")

    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_SERVICE_ROLE_KEY is missing. The publishable key is not enough for reliable backend job persistence."
        )

    if _SUPABASE_CLIENT is None:
        _SUPABASE_CLIENT = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    return _SUPABASE_CLIENT


def sanitize_output_name(filename: Optional[str]) -> str:
    """Create a safe base filename for generated assets."""
    raw_name = os.path.splitext(os.path.basename(filename or "image"))[0]
    cleaned_name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name).strip("_")
    return cleaned_name or "image"


def normalize_extension(filename: Optional[str]) -> str:
    """Return a safe file extension for uploaded inputs."""
    extension = os.path.splitext(filename or "")[1].lower()
    if extension in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        return extension
    return ".png"


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
    symbols: Literal["numbers", "letters", "custom"],
    min_area: int,
    resolution: int,
    symbol_size: float,
    prefill_dark: int,
    mode_filter: int,
    no_bilateral: bool,
    force_colors: Optional[str],
) -> Config:
    """Build a generator config from submitted form values."""
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


def serialize_config(
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
) -> dict:
    """Keep the submitted job parameters for debugging and replay."""
    return {
        "colors": colors,
        "difficulty": difficulty,
        "symbols": symbols,
        "min_area": min_area,
        "resolution": resolution,
        "symbol_size": symbol_size,
        "prefill_dark": prefill_dark,
        "mode_filter": mode_filter,
        "no_bilateral": no_bilateral,
        "force_colors": force_colors,
    }


def encode_png(image, label: str) -> bytes:
    """Encode a BGR image as PNG bytes."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Failed to encode {label} image")
    return buffer.tobytes()


def storage_client():
    """Return the Supabase storage bucket client."""
    return get_supabase().storage.from_(SUPABASE_STORAGE_BUCKET)


def upload_bytes_to_storage(path: str, data: bytes, content_type: str) -> None:
    """Upload raw bytes into Supabase Storage."""
    storage_client().upload(
        path=path,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )


def download_bytes_from_storage(path: str) -> bytes:
    """Download a file from Supabase Storage."""
    return storage_client().download(path)


def fetch_job(job_id: str) -> dict:
    """Load one job row from Supabase."""
    response = get_supabase().table(SUPABASE_JOBS_TABLE).select("*").eq("id", job_id).limit(1).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Job not found")
    return response.data[0]


def update_job(job_id: str, values: dict) -> dict:
    """Update one job row in Supabase."""
    values = {**values, "updated_at": format_datetime(utc_now())}
    response = get_supabase().table(SUPABASE_JOBS_TABLE).update(values).eq("id", job_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Job not found")
    return response.data[0]


def is_job_stale(job: dict) -> bool:
    """Check whether a processing job looks abandoned."""
    reference = parse_datetime(job.get("last_heartbeat_at")) or parse_datetime(job.get("updated_at"))
    if reference is None:
        return True
    return utc_now() - reference > timedelta(seconds=JOB_STALE_SECONDS)


def mark_job_running(job_id: str) -> bool:
    """Track locally scheduled jobs to avoid duplicate work on the same process."""
    with _ACTIVE_JOBS_LOCK:
        if job_id in _ACTIVE_JOBS:
            return False
        _ACTIVE_JOBS.add(job_id)
        return True


def unmark_job_running(job_id: str) -> None:
    """Remove a job from the local active set."""
    with _ACTIVE_JOBS_LOCK:
        _ACTIVE_JOBS.discard(job_id)


def heartbeat_loop(job_id: str, stop_event: threading.Event) -> None:
    """Refresh processing heartbeat while a job is running."""
    while not stop_event.wait(JOB_HEARTBEAT_SECONDS):
        try:
            update_job(job_id, {"last_heartbeat_at": format_datetime(utc_now())})
        except Exception:
            return


def process_job(job_id: str) -> None:
    """Run one queued job and persist all outputs to Supabase."""
    if not mark_job_running(job_id):
        return

    stop_event = threading.Event()
    heartbeat_thread: Optional[threading.Thread] = None

    try:
        job = fetch_job(job_id)
        if job["status"] == "completed":
            return

        update_job(
            job_id,
            {
                "status": "processing",
                "error": None,
                "completed_at": None,
                "last_heartbeat_at": format_datetime(utc_now()),
            },
        )

        heartbeat_thread = threading.Thread(target=heartbeat_loop, args=(job_id, stop_event), daemon=True)
        heartbeat_thread.start()

        input_bytes = download_bytes_from_storage(job["input_storage_path"])
        config_payload = job.get("config") or {}
        config = build_config(
            colors=int(config_payload.get("colors", 16)),
            difficulty=int(config_payload.get("difficulty", 5)),
            symbols=config_payload.get("symbols", "numbers"),
            min_area=int(config_payload.get("min_area", 50)),
            resolution=int(config_payload.get("resolution", 1400)),
            symbol_size=float(config_payload.get("symbol_size", 0.5)),
            prefill_dark=int(config_payload.get("prefill_dark", 500)),
            mode_filter=int(config_payload.get("mode_filter", 5)),
            no_bilateral=bool(config_payload.get("no_bilateral", False)),
            force_colors=config_payload.get("force_colors"),
        )

        generator = MysteryColoringGenerator(config)
        loaded_image = generator.load_image_bytes(input_bytes, job["source_filename"])
        outputs = generator.generate_images(loaded_image)

        combined_bytes = encode_png(outputs["combined"], "combined")
        preview_bytes = encode_png(outputs["preview"], "preview")

        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{job['output_name']}_combined.png", combined_bytes)
            archive.writestr(f"{job['output_name']}_preview.png", preview_bytes)
        archive_bytes = archive_buffer.getvalue()

        combined_storage_path = f"jobs/{job_id}/{job['output_name']}_combined.png"
        preview_storage_path = f"jobs/{job_id}/{job['output_name']}_preview.png"
        archive_storage_path = f"jobs/{job_id}/{job['output_name']}_outputs.zip"

        upload_bytes_to_storage(combined_storage_path, combined_bytes, "image/png")
        upload_bytes_to_storage(preview_storage_path, preview_bytes, "image/png")
        upload_bytes_to_storage(archive_storage_path, archive_bytes, "application/zip")

        update_job(
            job_id,
            {
                "status": "completed",
                "combined_storage_path": combined_storage_path,
                "preview_storage_path": preview_storage_path,
                "archive_storage_path": archive_storage_path,
                "completed_at": format_datetime(utc_now()),
                "last_heartbeat_at": format_datetime(utc_now()),
                "error": None,
            },
        )
    except Exception as exc:
        try:
            update_job(
                job_id,
                {
                    "status": "failed",
                    "error": str(exc),
                    "completed_at": format_datetime(utc_now()),
                },
            )
        except Exception:
            pass
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)
        unmark_job_running(job_id)


def maybe_schedule_job(job: dict) -> None:
    """Schedule work for queued or stale jobs."""
    if job["status"] == "queued":
        JOB_EXECUTOR.submit(process_job, job["id"])
    elif job["status"] == "processing" and is_job_stale(job):
        JOB_EXECUTOR.submit(process_job, job["id"])


def serialize_job(job: dict, request: Request) -> dict:
    """Convert the Supabase row into an API response payload."""
    payload = {
        "job_id": job["id"],
        "status": job["status"],
        "output_name": job["output_name"],
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "completed_at": job.get("completed_at"),
        "source_filename": job.get("source_filename"),
        "status_url": str(request.url_for("get_job_status", job_id=job["id"])),
    }

    if job["status"] == "completed":
        payload["result"] = {
            "combined_url": str(request.url_for("download_job_file", job_id=job["id"], file_kind="combined")),
            "preview_url": str(request.url_for("download_job_file", job_id=job["id"], file_kind="preview")),
            "archive_url": str(request.url_for("download_job_archive", job_id=job["id"])),
        }

    if job["status"] == "failed":
        payload["error"] = job.get("error")

    return payload


@app.get("/")
def root() -> dict:
    """Basic info endpoint."""
    return {
        "name": "Mystery Coloring API",
        "docs": "/docs",
        "health": "/health",
        "generate": "/generate",
        "job_status": "/jobs/{job_id}",
        "job_archive": "/jobs/{job_id}/download",
        "job_file": "/jobs/{job_id}/files/{file_kind}",
    }


@app.head("/")
def root_head() -> Response:
    """HEAD response for platform probes."""
    return Response(status_code=200)


@app.get("/health")
def health() -> dict:
    """Health endpoint for Render."""
    return {
        "status": "ok",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
        "storage_bucket": SUPABASE_STORAGE_BUCKET,
    }


@app.head("/health")
def health_head() -> Response:
    """HEAD health response for platform probes."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(
    request: Request,
    image: UploadFile = File(...),
    output_name: Optional[str] = Form(None),
    colors: int = Form(16),
    difficulty: int = Form(5),
    symbols: Literal["numbers", "letters", "custom"] = Form("numbers"),
    min_area: int = Form(50),
    resolution: int = Form(1400),
    symbol_size: float = Form(0.5),
    prefill_dark: int = Form(500),
    mode_filter: int = Form(5),
    no_bilateral: bool = Form(False),
    force_colors: Optional[str] = Form(None),
):
    """Create a generation job and return immediately with a job id."""
    if not image.filename:
        raise HTTPException(status_code=400, detail="A filename is required for the uploaded image")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")

    try:
        get_supabase()
        build_config(
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
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job_id = uuid4().hex
    base_name = sanitize_output_name(output_name or image.filename)
    input_extension = normalize_extension(image.filename)
    input_storage_path = f"jobs/{job_id}/input{input_extension}"

    config_payload = serialize_config(
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

    created_at = format_datetime(utc_now())

    try:
        upload_bytes_to_storage(input_storage_path, image_bytes, image.content_type or "application/octet-stream")
        response = get_supabase().table(SUPABASE_JOBS_TABLE).insert(
            {
                "id": job_id,
                "status": "queued",
                "source_filename": image.filename,
                "output_name": base_name,
                "input_storage_path": input_storage_path,
                "combined_storage_path": None,
                "preview_storage_path": None,
                "archive_storage_path": None,
                "error": None,
                "config": config_payload,
                "created_at": created_at,
                "updated_at": created_at,
                "completed_at": None,
                "last_heartbeat_at": None,
            }
        ).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create Supabase job: {exc}") from exc

    if not response.data:
        raise HTTPException(status_code=500, detail="Supabase did not return the created job")

    job = response.data[0]
    maybe_schedule_job(job)

    payload = serialize_job(job, request)
    payload["message"] = "Job created. Poll status_url until status is completed."

    headers = {
        "Location": str(request.url_for("get_job_status", job_id=job_id)),
        "Retry-After": "5",
    }
    return JSONResponse(status_code=202, content=payload, headers=headers)


@app.get("/jobs/{job_id}", name="get_job_status")
def get_job_status(job_id: str, request: Request) -> dict:
    """Return the status of a generation job."""
    job = fetch_job(job_id)
    maybe_schedule_job(job)
    refreshed_job = fetch_job(job_id)
    return serialize_job(refreshed_job, request)


@app.get("/jobs/{job_id}/download", name="download_job_archive")
def download_job_archive(job_id: str) -> StreamingResponse:
    """Download the generated ZIP once the job is complete."""
    job = fetch_job(job_id)

    if job["status"] in {"queued", "processing"}:
        maybe_schedule_job(job)
        raise HTTPException(status_code=409, detail=f"Job is not ready yet (status: {job['status']})")

    if job["status"] == "failed":
        raise HTTPException(status_code=409, detail=job.get("error") or "Job failed")

    archive_path = job.get("archive_storage_path")
    if not archive_path:
        raise HTTPException(status_code=500, detail="Job completed but archive path is missing")

    archive_bytes = download_bytes_from_storage(archive_path)
    headers = {
        "Content-Disposition": f'attachment; filename="{job["output_name"]}_outputs.zip"'
    }
    return StreamingResponse(io.BytesIO(archive_bytes), media_type="application/zip", headers=headers)


@app.get("/jobs/{job_id}/files/{file_kind}", name="download_job_file")
def download_job_file(job_id: str, file_kind: str) -> StreamingResponse:
    """Download one generated image from Supabase Storage."""
    job = fetch_job(job_id)

    if job["status"] in {"queued", "processing"}:
        maybe_schedule_job(job)
        raise HTTPException(status_code=409, detail=f"Job is not ready yet (status: {job['status']})")

    if job["status"] == "failed":
        raise HTTPException(status_code=409, detail=job.get("error") or "Job failed")

    if file_kind == "combined":
        storage_path = job.get("combined_storage_path")
        filename = f"{job['output_name']}_combined.png"
    elif file_kind == "preview":
        storage_path = job.get("preview_storage_path")
        filename = f"{job['output_name']}_preview.png"
    else:
        raise HTTPException(status_code=404, detail="Unknown file kind")

    if not storage_path:
        raise HTTPException(status_code=500, detail="Job completed but requested file path is missing")

    file_bytes = download_bytes_from_storage(storage_path)
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"'
    }
    return StreamingResponse(io.BytesIO(file_bytes), media_type="image/png", headers=headers)
