from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gmes_plot.domain.models import Dataset


ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "x": ("x", "easting", "east", "longitude", "lon", "distance", "position", "station", "里程", "距离", "位置", "东坐标", "经度"),
    "y": ("y", "northing", "north", "latitude", "lat", "depth", "elevation", "height", "深度", "高程", "北坐标", "纬度"),
    "z": ("z", "depth", "elevation", "height", "altitude", "time", "高程", "深度", "标高", "时间"),
    "value": ("value", "v", "data", "amplitude", "resistivity", "velocity", "density", "magnetic", "gravity", "值", "数值", "振幅", "电阻率", "速度", "密度", "磁异常", "重力异常"),
}


def suggest_roles(names: list[str]) -> dict[str, str]:
    """Suggest unambiguous field roles, then fall back to conventional column order."""
    normalized = [re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", name).casefold() for name in names]
    roles: dict[str, str] = {}
    used: set[str] = set()
    for role in ("x", "y", "z", "value"):
        aliases = ROLE_ALIASES[role]
        match = next(
            (names[index] for index, item in enumerate(normalized)
             if names[index] not in used and any(item == alias.casefold() for alias in aliases)),
            None,
        )
        if match is not None:
            roles[role] = match
            used.add(match)

    # A bare 3-column profile is XYV; 4+ columns conventionally begin XYZV.
    fallback_order = ("x", "y", "value") if len(names) == 3 else ("x", "y", "z", "value")
    for index, role in enumerate(fallback_order):
        if index >= len(names) or role in roles:
            continue
        candidate = names[index]
        if candidate not in used:
            roles[role] = candidate
            used.add(candidate)
    return roles


@dataclass(frozen=True, slots=True)
class ParseOptions:
    encoding: str = "auto"
    delimiter: str | None = None
    header: bool | None = None
    skip_rows: int = 0
    comment_prefix: str = "#"


@dataclass(slots=True)
class TablePreview:
    path: Path
    encoding: str
    delimiter: str | None
    has_header: bool
    names: list[str]
    rows: list[list[str]]
    warnings: list[str]


def _decode(data: bytes, requested: str) -> tuple[str, str]:
    encodings = [requested] if requested != "auto" else ["utf-8-sig", "utf-8", "gb18030"]
    for encoding in encodings:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码，请手动指定编码")


def _detect_delimiter(lines: list[str]) -> str | None:
    sample = "\n".join(lines[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return None


def _split(line: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return re.split(r"\s+", line.strip())
    return next(csv.reader([line], delimiter=delimiter))


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def preview_table(path: str | Path, options: ParseOptions = ParseOptions(), limit: int = 50) -> TablePreview:
    file_path = Path(path)
    text, encoding = _decode(file_path.read_bytes(), options.encoding)
    lines = [line for line in text.splitlines()[options.skip_rows:] if line.strip() and not line.lstrip().startswith(options.comment_prefix)]
    if not lines:
        raise ValueError("文件中没有可读取的数据行")
    delimiter = options.delimiter if options.delimiter is not None else _detect_delimiter(lines)
    first = _split(lines[0], delimiter)
    has_header = options.header if options.header is not None else not all(_is_number(item.strip()) for item in first)
    names = [item.strip() or f"Column_{i + 1}" for i, item in enumerate(first)] if has_header else [f"Column_{i + 1}" for i in range(len(first))]
    data_lines = lines[1:] if has_header else lines
    rows = [_split(line, delimiter) for line in data_lines[:limit]]
    warnings: list[str] = []
    bad = [i for i, row in enumerate(rows, start=2 if has_header else 1) if len(row) != len(names)]
    if bad:
        warnings.append(f"预览中发现列数不一致的行: {bad[:5]}")
    return TablePreview(file_path, encoding, delimiter, has_header, names, rows, warnings)


def load_dataset(
    path: str | Path,
    roles: dict[str, str],
    options: ParseOptions = ParseOptions(),
    name: str | None = None,
) -> Dataset:
    file_path = Path(path)
    preview = preview_table(file_path, options)
    text, _ = _decode(file_path.read_bytes(), preview.encoding)
    lines = [line for line in text.splitlines()[options.skip_rows:] if line.strip() and not line.lstrip().startswith(options.comment_prefix)]
    if preview.has_header:
        lines = lines[1:]
    columns: dict[str, list[float | str]] = {column: [] for column in preview.names}
    for line_number, line in enumerate(lines, start=2 if preview.has_header else 1):
        row = _split(line, preview.delimiter)
        if len(row) != len(preview.names):
            raise ValueError(f"第 {line_number} 行列数为 {len(row)}，预期 {len(preview.names)}")
        for column, value in zip(preview.names, row):
            stripped = value.strip()
            columns[column].append(float(stripped) if _is_number(stripped) else stripped)
    arrays: dict[str, np.ndarray] = {}
    required = set(roles.values())
    for column, values in columns.items():
        if column in required:
            try:
                arrays[column] = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"字段 {column} 被映射为数值角色，但包含非数值") from exc
        else:
            arrays[column] = np.asarray(values)
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return Dataset(
        name=name or file_path.stem,
        columns=arrays,
        roles=dict(roles),
        source_path=str(file_path.resolve()),
        source_hash=digest,
    )

