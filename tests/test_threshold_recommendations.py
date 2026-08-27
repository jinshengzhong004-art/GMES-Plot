import numpy as np

from gmes_plot.services.thresholds import recommend_anomaly_thresholds


def test_resistivity_recommendation_uses_log_space():
    values = np.logspace(0, 4, 101)
    result = recommend_anomaly_thresholds(values, "resistivity")
    assert result.transform == "log10 P10/P90"
    assert 1 < result.low < result.high < 10000


def test_gravity_recommendation_is_robust_to_extreme_outlier():
    values = np.r_[np.linspace(-2, 2, 100), 1e9]
    result = recommend_anomaly_thresholds(values, "gravity")
    assert result.high < 10
    assert result.low > -10


def test_seismic_amplitude_is_symmetric_around_robust_center():
    values = np.linspace(-5, 5, 101)
    result = recommend_anomaly_thresholds(values, "seismic_amplitude")
    assert np.isclose(result.low, -result.high)

