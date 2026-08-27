from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gmes_plot import __version__
from gmes_plot.domain.models import Dataset, GridResult, Project, SpatialFilter, Threshold


def layer_provenance(
    dataset: Dataset | None,
    grid: GridResult | None,
    *,
    threshold: Threshold,
    spatial_filter: SpatialFilter,
    cmap: str,
    style: dict[str, Any],
    view_kind: str,
    physical_method: str,
) -> dict[str, Any]:
    """Create a serialisable scientific lineage snapshot for one view."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "application": {"name": "GMES-Plot", "version": __version__},
        "view_kind": view_kind,
        "source_data": None if dataset is None else {
            "id": dataset.id,
            "name": dataset.name,
            "source_path": dataset.source_path,
            "source_hash": dataset.source_hash,
            "row_count": dataset.row_count,
            "field_mapping": dict(dataset.roles),
            "units": dict(dataset.units),
            "crs": dataset.crs,
            "parent_id": dataset.parent_id,
            "derivation": dataset.derivation,
            "created_at": dataset.created_at,
        },
        "derived_grid": None if grid is None else {
            "id": grid.id,
            "name": grid.name,
            "method": grid.method,
            "bounds": list(grid.spec.bounds),
            "shape": list(grid.spec.shape),
            "cell_count": grid.spec.cell_count,
            "parameters": dict(grid.metadata),
            "has_uncertainty": grid.variance is not None,
        },
        "display": {
            "spatial_filter_priority": "XYZ before Value",
            "spatial_filter": {name: getattr(spatial_filter, name) for name in ("x", "y", "z")},
            "value_threshold": {"mode": threshold.mode.value, "lower": threshold.lower, "upper": threshold.upper},
            "physical_method": physical_method,
            "color_scheme": cmap,
            "color_scheme_version": style.get("color_scheme_version", "built-in"),
            "style": dict(style),
        },
    }


def provenance_markdown(record: dict[str, Any]) -> str:
    source = record.get("source_data") or {}
    grid = record.get("derived_grid") or {}
    display = record.get("display") or {}
    return "\n".join([
        "# GMES-Plot 科研复现报告",
        "",
        f"- 生成时间：{record.get('generated_at', '—')}",
        f"- 软件版本：{record.get('application', {}).get('version', '—')}",
        f"- 图形类型：{record.get('view_kind', '—')}",
        f"- 来源数据：{source.get('name', '—')}（{source.get('id', '—')}）",
        f"- 来源文件：{source.get('source_path') or '工程内便携数据/未记录'}",
        f"- SHA-256：{source.get('source_hash') or '未记录'}",
        f"- 字段映射：{source.get('field_mapping', {})}",
        f"- 坐标系：{source.get('crs') or '未设置'}",
        f"- 单位：{source.get('units', {})}",
        f"- 插值方法：{grid.get('method', '未使用')}",
        f"- 网格形状：{grid.get('shape', '—')}；范围：{grid.get('bounds', '—')}",
        f"- 插值参数：{grid.get('parameters', {})}",
        f"- XYZ显示裁剪：{display.get('spatial_filter', {})}",
        f"- Value显示阈值：{display.get('value_threshold', {})}",
        f"- 物探语义：{display.get('physical_method', 'generic')}",
        f"- 颜色方案：{display.get('color_scheme', '—')}（{display.get('color_scheme_version', '—')}）",
        "",
        "> 阈值与空间裁剪均属于显示层；如需改变插值输入，必须创建派生数据集。",
    ]) + "\n"


def write_companion_report(image_path: str | Path, record: dict[str, Any]) -> Path:
    target = Path(image_path).with_suffix(".reproducibility.md")
    target.write_text(provenance_markdown(record), encoding="utf-8")
    return target


def publication_checks(figure, project: Project, record: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    source = record.get("source_data") or {}
    if not source.get("units"):
        findings.append({"level": "warning", "message": "来源数据未设置单位。"})
    if not source.get("crs"):
        findings.append({"level": "info", "message": "来源数据未设置坐标系；局部坐标数据可忽略。"})
    if any(float(text.get_fontsize()) < 7 for axis in figure.axes for text in [axis.title, axis.xaxis.label, axis.yaxis.label]):
        findings.append({"level": "warning", "message": "存在小于 7 pt 的标题或轴文字，印刷后可能难以辨认。"})
    for axis in figure.axes:
        if getattr(axis, "_gmes_colorbar", False):
            x, y, width, height = axis.get_position().bounds
            if x < 0 or y < 0 or x + width > 1 or y + height > 1:
                findings.append({"level": "high", "message": "色标超出页面范围。"})
    grid = record.get("derived_grid") or {}
    if grid and grid.get("id") not in project.reports:
        findings.append({"level": "warning", "message": "当前插值结果尚未生成质量报告。"})
    if source.get("parent_id") and not source.get("derivation"):
        findings.append({"level": "high", "message": "派生数据集缺少处理参数记录。"})
    if not findings:
        findings.append({"level": "pass", "message": "未发现阻断导出的科研排版问题。"})
    return findings

