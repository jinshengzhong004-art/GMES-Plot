from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from gmes_plot import __version__
from gmes_plot.domain.models import Dataset, GridResult, GridSpec, Project, TerrainProfile

FORMAT_VERSION = "1.0"


def _dataset_meta(dataset: Dataset) -> dict[str, Any]:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "roles": dataset.roles,
        "units": dataset.units,
        "crs": dataset.crs,
        "source_path": dataset.source_path,
        "source_hash": dataset.source_hash,
        "parent_id": dataset.parent_id,
        "derivation": dataset.derivation,
        "created_at": dataset.created_at,
        "columns": list(dataset.columns),
    }


def save_project(project: Project, path: str | Path) -> None:
    target = Path(path)
    if target.suffix.lower() != ".gpproj":
        target = target.with_suffix(".gpproj")
    temporary = target.with_suffix(target.suffix + ".tmp")
    manifest = {
        "format_version": FORMAT_VERSION,
        "application_version": __version__,
        "id": project.id,
        "name": project.name,
        "datasets": [_dataset_meta(item) for item in project.datasets.values()],
        "grids": [
            {
                "id": grid.id,
                "name": grid.name,
                "bounds": grid.spec.bounds,
                "shape": grid.spec.shape,
                "method": grid.method,
                "source_dataset_id": grid.source_dataset_id,
                "metadata": grid.metadata,
                "has_variance": grid.variance is not None,
            }
            for grid in project.grids.values()
        ],
        "terrains": [
            {
                "id": terrain.id,
                "name": terrain.name,
                "source_path": terrain.source_path,
                "source_hash": terrain.source_hash,
                "position_unit": terrain.position_unit,
                "elevation_unit": terrain.elevation_unit,
            }
            for terrain in project.terrains.values()
        ],
        "pages": project.pages,
        "styles": project.styles,
        "colormaps": project.colormaps,
        "reports": project.reports,
        "constraints": project.constraints,
        "standards_profile": project.standards_profile,
    }
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for dataset in project.datasets.values():
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **dataset.columns)
            archive.writestr(f"datasets/{dataset.id}.npz", buffer.getvalue())
        for grid in project.grids.values():
            buffer = io.BytesIO()
            payload = {"values": grid.values}
            if grid.variance is not None:
                payload["variance"] = grid.variance
            np.savez_compressed(buffer, **payload)
            archive.writestr(f"grids/{grid.id}.npz", buffer.getvalue())
        for terrain in project.terrains.values():
            buffer = io.BytesIO()
            np.savez_compressed(buffer, position=terrain.position, elevation=terrain.elevation)
            archive.writestr(f"terrains/{terrain.id}.npz", buffer.getvalue())
    temporary.replace(target)
    project.project_path = target
    project.dirty = False


def load_project(path: str | Path) -> Project:
    source = Path(path)
    with zipfile.ZipFile(source, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["format_version"].split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise ValueError(f"不支持的工程主版本: {manifest['format_version']}")
        project = Project(name=manifest["name"], id=manifest["id"])
        for meta in manifest.get("datasets", []):
            with np.load(io.BytesIO(archive.read(f"datasets/{meta['id']}.npz")), allow_pickle=False) as payload:
                columns = {name: payload[name] for name in payload.files}
            dataset = Dataset(
                id=meta["id"], name=meta["name"], columns=columns, roles=meta["roles"],
                source_path=meta.get("source_path"), source_hash=meta.get("source_hash"),
                units=meta.get("units", {}), crs=meta.get("crs"), parent_id=meta.get("parent_id"),
                derivation=meta.get("derivation"), created_at=meta.get("created_at", ""),
            )
            project.datasets[dataset.id] = dataset
        for meta in manifest.get("grids", []):
            with np.load(io.BytesIO(archive.read(f"grids/{meta['id']}.npz")), allow_pickle=False) as payload:
                values = payload["values"]
                variance = payload["variance"] if "variance" in payload.files else None
            grid = GridResult(
                id=meta["id"], name=meta["name"], spec=GridSpec(tuple(meta["bounds"]), tuple(meta["shape"])),
                values=values, variance=variance, method=meta["method"],
                source_dataset_id=meta["source_dataset_id"], metadata=meta.get("metadata", {}),
            )
            project.grids[grid.id] = grid
        for meta in manifest.get("terrains", []):
            with np.load(io.BytesIO(archive.read(f"terrains/{meta['id']}.npz")), allow_pickle=False) as payload:
                position = payload["position"]
                elevation = payload["elevation"]
            terrain = TerrainProfile(
                id=meta["id"], name=meta["name"], position=position, elevation=elevation,
                source_path=meta.get("source_path"), source_hash=meta.get("source_hash"),
                position_unit=meta.get("position_unit", "m"), elevation_unit=meta.get("elevation_unit", "m"),
            )
            project.terrains[terrain.id] = terrain
        project.pages = manifest.get("pages", [])
        project.styles = manifest.get("styles", {})
        project.colormaps = manifest.get("colormaps", {})
        project.reports = manifest.get("reports", {})
        project.constraints = manifest.get("constraints", {"boreholes": [], "geology": []})
        project.standards_profile = manifest.get("standards_profile", {})
        project.project_path = source
        project.dirty = False
        return project

