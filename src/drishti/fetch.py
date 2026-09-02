"""Fetch a Sentinel-2 L2A scene from the Copernicus Data Space Ecosystem.

Uses the Sentinel Hub Process API (OAuth client-credentials) hosted at
sh.dataspace.copernicus.eu. Given a bounding box, a date range and a maximum
cloud cover, this returns the newest matching Sentinel-2 L2A scene as a single
multi-band Cloud-Optimised GeoTIFF in surface reflectance.

Output bands, in order, all on the native 10 m grid (UTM, scene's zone):
    1  B02  490 nm   (10 m native)
    2  B03  560 nm   (10 m native)
    3  B04  665 nm   (10 m native)
    4  B08  842 nm   (10 m native)
    5  B05  705 nm   (20 m native, resampled to 10 m)
    6  B06  740 nm   (20 m native, resampled to 10 m)
    7  B07  783 nm   (20 m native, resampled to 10 m)
    8  B8A  865 nm   (20 m native, resampled to 10 m)
    9  B11 1610 nm   (20 m native, resampled to 10 m)
    10 B12 2190 nm   (20 m native, resampled to 10 m)

Values are surface reflectance as float32 in roughly [0, 1] (L2A boundary
reflectance; can slightly exceed 1 over bright targets). Never 8-bit stretched.

A sidecar ``<output>.json`` is written next to the GeoTIFF with the scene id,
acquisition datetime, cloud cover and CRS.

Credentials are read from ``.env`` via python-dotenv and are never hardcoded:
    CDSE_CLIENT_ID
    CDSE_CLIENT_SECRET
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import rasterio
import requests
from dotenv import load_dotenv
from rasterio.io import MemoryFile

# --- Band configuration ------------------------------------------------------

BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
BANDS = BANDS_10M + BANDS_20M

NATIVE_RES_M = 10.0
MAX_PROCESS_DIM = 2500  # Sentinel Hub Process API hard limit per request (px)

DEFAULT_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
DEFAULT_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
DEFAULT_CATALOG_URL = (
    "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
)


# --- Data structures --------------------------------------------------------


@dataclass
class SceneInfo:
    """Metadata for the selected Sentinel-2 L2A scene (goes to the sidecar)."""

    scene_id: str
    acquired: str  # ISO 8601 UTC
    cloud_cover: float  # percent
    crs: str  # e.g. "EPSG:32643"
    bbox_utm: list[float]  # [minx, miny, maxx, maxy] in `crs`
    resolution_m: float
    bands: list[str]
    source: str = "Sentinel-2 L2A / Copernicus Data Space (Sentinel Hub Process API)"


# --- Auth ------------------------------------------------------------------


def _load_credentials() -> tuple[str, str]:
    load_dotenv()
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are not set. Copy .env.example "
            "to .env and fill them in (create a client at "
            "https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings)."
        )
    return client_id, client_secret


def _get_token(session: requests.Session) -> str:
    client_id, client_secret = _load_credentials()
    token_url = os.environ.get("CDSE_TOKEN_URL", DEFAULT_TOKEN_URL)
    resp = session.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


# --- Geometry helpers -----------------------------------------------------


def _utm_epsg_for_bbox(bbox_ll: list[float]) -> int:
    """Pick the UTM zone EPSG code for the centre of a lon/lat bbox."""
    lon = (bbox_ll[0] + bbox_ll[2]) / 2.0
    lat = (bbox_ll[1] + bbox_ll[3]) / 2.0
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _reproject_bbox(bbox_ll: list[float], dst_epsg: int) -> list[float]:
    """Reproject a lon/lat bbox to `dst_epsg` and snap it outward to the
    NATIVE_RES_M grid so the output pixel size is exactly NATIVE_RES_M and the
    grid aligns with the standard Sentinel-2 UTM grid (origin at 0)."""
    from rasterio.warp import transform_bounds

    minx, miny, maxx, maxy = transform_bounds(
        "EPSG:4326", f"EPSG:{dst_epsg}", *bbox_ll
    )
    r = NATIVE_RES_M
    return [
        math.floor(minx / r) * r,
        math.floor(miny / r) * r,
        math.ceil(maxx / r) * r,
        math.ceil(maxy / r) * r,
    ]


def _grid_size(bbox_utm: list[float], res_m: float) -> tuple[int, int]:
    width = int(round((bbox_utm[2] - bbox_utm[0]) / res_m))
    height = int(round((bbox_utm[3] - bbox_utm[1]) / res_m))
    return width, height


# --- Catalog: find the newest matching scene ----------------------------


def _find_scene(
    session: requests.Session,
    token: str,
    bbox_ll: list[float],
    start: str,
    end: str,
    max_cloud: float,
) -> dict:
    catalog_url = os.environ.get("CDSE_CATALOG_URL", DEFAULT_CATALOG_URL)
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox_ll,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 100,
        "filter": {
            "op": "<=",
            "args": [{"property": "eo:cloud_cover"}, max_cloud],
        },
        "filter-lang": "cql2-json",
        "fields": {
            "include": [
                "id",
                "properties.datetime",
                "properties.eo:cloud_cover",
                "properties.proj:epsg",
            ]
        },
    }
    resp = session.post(
        catalog_url,
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])

    # Defensive client-side filter + sort (do not rely on server ordering).
    def cloud(f: dict) -> float:
        return float(f["properties"].get("eo:cloud_cover", 100.0))

    features = [f for f in features if cloud(f) <= max_cloud]
    if not features:
        raise RuntimeError(
            f"No Sentinel-2 L2A scene over {bbox_ll} between {start} and {end} "
            f"with cloud cover <= {max_cloud}%."
        )
    features.sort(key=lambda f: f["properties"]["datetime"], reverse=True)
    return features[0]


# --- Process API: download the pixels --------------------------------------


def _evalscript() -> str:
    band_list = ", ".join(f'"{b}"' for b in BANDS)
    returns = ", ".join(f"s.{b}" for b in BANDS)
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: [{band_list}], units: "REFLECTANCE" }}],
    output: {{ bands: {len(BANDS)}, sampleType: "FLOAT32" }}
  }};
}}
function evaluatePixel(s) {{
  return [{returns}];
}}
"""


def _download_tiff(
    session: requests.Session,
    token: str,
    scene: dict,
    bbox_utm: list[float],
    dst_epsg: int,
    max_cloud: float,
) -> bytes:
    process_url = os.environ.get("CDSE_PROCESS_URL", DEFAULT_PROCESS_URL)
    width, height = _grid_size(bbox_utm, NATIVE_RES_M)
    if width > MAX_PROCESS_DIM or height > MAX_PROCESS_DIM:
        raise ValueError(
            f"Requested grid is {width}x{height} px at {NATIVE_RES_M} m; the "
            f"Process API caps a single request at {MAX_PROCESS_DIM} px per "
            "axis. Use a smaller bbox for now (tiled fetch comes later)."
        )

    # Narrow the time window to the acquisition day so the Process API returns
    # exactly the scene the catalog picked (mosaicking defaults to mostRecent).
    day = scene["properties"]["datetime"][:10]
    body = {
        "input": {
            "bounds": {
                "bbox": bbox_utm,
                "properties": {
                    "crs": f"http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{day}T00:00:00Z",
                            "to": f"{day}T23:59:59Z",
                        },
                        "maxCloudCoverage": max_cloud,
                    },
                    # 20 m bands are upsampled to the 10 m output grid here.
                    "processing": {"upsampling": "BILINEAR"},
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {"identifier": "default", "format": {"type": "image/tiff"}}
            ],
        },
        "evalscript": _evalscript(),
    }
    resp = session.post(
        process_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "image/tiff",
        },
        json=body,
        timeout=600,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Process API returned {resp.status_code}: {resp.text[:1000]}"
        )
    return resp.content


# --- COG writing ----------------------------------------------------------


def _write_cog(raw_tiff: bytes, out_path: Path, scene: SceneInfo) -> None:
    with MemoryFile(raw_tiff) as mem, mem.open() as src:
        data = src.read()  # (bands, H, W), float32
        profile = src.profile.copy()

        # TODO(DSen2): band sharpening for the 20 m bands (B05 B06 B07 B8A B11
        # B12) goes here. Right now they carry a bilinear upsample from the
        # Process API. Replace with a DSen2-style CNN that sharpens the 20 m
        # bands using the 10 m bands (B02 B03 B04 B08) as guidance, in
        # reflectance, before the COG is written. Keep band order unchanged.

        profile.update(
            driver="COG",
            dtype="float32",
            compress="deflate",
            predictor=3,
            blocksize=512,
            overview_resampling="bilinear",
            BIGTIFF="IF_SAFER",
        )
        profile.pop("tiled", None)
        profile.pop("interleave", None)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
            for i, name in enumerate(scene.bands, start=1):
                dst.set_band_description(i, name)
            dst.update_tags(
                scene_id=scene.scene_id,
                acquired=scene.acquired,
                cloud_cover=str(scene.cloud_cover),
                source=scene.source,
            )


# --- Public API ---------------------------------------------------------


def fetch_scene(
    bbox: list[float],
    start: str,
    end: str,
    max_cloud: float,
    out: str | Path,
) -> SceneInfo:
    """Fetch the newest matching Sentinel-2 L2A scene as a multi-band COG.

    Parameters
    ----------
    bbox : [min_lon, min_lat, max_lon, max_lat] in EPSG:4326 degrees.
    start, end : "YYYY-MM-DD" (inclusive), UTC.
    max_cloud : maximum scene cloud cover, percent (0-100).
    out : path to the output GeoTIFF. A "<out>.json" sidecar is written too.

    Returns
    -------
    SceneInfo for the selected scene.
    """
    if len(bbox) != 4:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat].")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dst_epsg = _utm_epsg_for_bbox(bbox)
    bbox_utm = _reproject_bbox(bbox, dst_epsg)

    session = requests.Session()
    token = _get_token(session)
    feature = _find_scene(session, token, bbox, start, end, max_cloud)

    scene = SceneInfo(
        scene_id=feature["id"],
        acquired=feature["properties"]["datetime"],
        cloud_cover=round(float(feature["properties"]["eo:cloud_cover"]), 3),
        crs=f"EPSG:{dst_epsg}",
        bbox_utm=[round(v, 3) for v in bbox_utm],
        resolution_m=NATIVE_RES_M,
        bands=BANDS,
    )

    raw_tiff = _download_tiff(
        session, token, feature, bbox_utm, dst_epsg, max_cloud
    )
    _write_cog(raw_tiff, out_path, scene)

    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    payload = asdict(scene)
    payload["fetched_at"] = datetime.now(timezone.utc).isoformat()
    payload["native_proj_epsg"] = feature["properties"].get("proj:epsg")
    sidecar.write_text(json.dumps(payload, indent=2))

    return scene
