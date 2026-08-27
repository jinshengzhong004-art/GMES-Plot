from __future__ import annotations

from dataclasses import dataclass

import numpy as np


METHOD_LABELS = {
    "generic": "通用数值",
    "gravity": "重力/密度异常",
    "magnetic": "磁法异常",
    "resistivity": "电阻率",
    "conductivity": "电导率/电磁",
    "chargeability": "极化率/充电率",
    "seismic_velocity": "地震/声波速度",
    "seismic_amplitude": "地震振幅",
}


@dataclass(frozen=True, slots=True)
class AnomalyRecommendation:
    method: str
    low: float
    high: float
    transform: str
    explanation: str


def _finite(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result = result[np.isfinite(result)]
    if result.size < 3:
        raise ValueError("异常阈值推荐至少需要3个有效数值")
    return result


def _robust_limits(values: np.ndarray, factor: float = 2.5) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale == 0:
        return float(np.quantile(values, .1)), float(np.quantile(values, .9))
    return median - factor * scale, median + factor * scale


def recommend_anomaly_thresholds(values: np.ndarray, method: str) -> AnomalyRecommendation:
    data = _finite(values)
    if method == "resistivity":
        positive = data[data > 0]
        if positive.size < 3:
            raise ValueError("电阻率对数阈值要求至少3个正值")
        log_values = np.log10(positive)
        low, high = np.quantile(log_values, [.1, .9])
        return AnomalyRecommendation(method, float(10**low), float(10**high), "log10 P10/P90", "电阻率跨数量级，按log10空间的P10/P90推荐低阻和高阻异常。")
    if method in ("gravity", "magnetic"):
        low, high = _robust_limits(data, 2.5)
        return AnomalyRecommendation(method, low, high, "median ± 2.5×1.4826MAD", "重磁异常采用稳健背景与MAD尺度，降低极端点对阈值的影响。")
    if method == "seismic_amplitude":
        center = float(np.median(data))
        amplitude = float(np.quantile(np.abs(data - center), .9))
        return AnomalyRecommendation(method, center - amplitude, center + amplitude, "median ± P90(|residual|)", "振幅按相对稳健中心的绝对偏差识别正、负强异常。")
    if method == "seismic_velocity":
        low, high = _robust_limits(data, 2.0)
        return AnomalyRecommendation(method, low, high, "median ± 2×1.4826MAD", "速度异常采用稳健中心，分别给出低速和高速建议。")
    if method == "conductivity":
        low, high = np.quantile(data, [.1, .9])
        return AnomalyRecommendation(method, float(low), float(high), "P10/P90", "高电导对应导电异常；若输入为电阻率，应改选“电阻率”方法。")
    if method == "chargeability":
        low, high = np.quantile(data, [.1, .9])
        return AnomalyRecommendation(method, float(low), float(high), "P10/P90", "极化率按分位数给出低、高极化候选区。")
    low, high = np.quantile(data, [.1, .9])
    return AnomalyRecommendation(method, float(low), float(high), "P10/P90", "通用数值按P10/P90给出候选异常，需结合专业背景确认。")

