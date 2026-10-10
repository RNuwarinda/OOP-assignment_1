import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cosine as scipy_cosine_distance

from src.rainfall import CropRule, Region
from src.rainfall_analysis import (
    NASA_POWER_LOCATIONS,
    analyze_rainy_seasons,
    classify_months,
    cosine_similarity,
    create_crop_rules,
    create_regions,
    detect_rainfall_peaks,
    download_nasa_power_rainfall,
    farmer_advisory,
    pairwise_region_metrics,
    plot_nasa_monthly_boxplots,
    region_summary,
    run_analysis,
)


def test_region_stores_array_and_computes_rainfall_summaries():
    region = Region("Example", np.arange(12))

    assert isinstance(region.rainfall_mm, np.ndarray)
    assert region.annual_total() == pytest.approx(66)
    assert region.mean_monthly_rainfall() == pytest.approx(5.5)
    assert region.wettest_month() == ("Dec", 11.0)
    assert region.driest_month() == ("Jan", 0.0)
    assert region.coefficient_of_variation() == pytest.approx(
        np.std(np.arange(12)) / np.mean(np.arange(12))
    )


@pytest.mark.parametrize(
    "rainfall",
    [
        [1] * 11,
        [-1] + [1] * 11,
        [np.nan] + [1] * 11,
        [[1] * 12],
    ],
)
def test_region_rejects_invalid_rainfall(rainfall):
    with pytest.raises(ValueError):
        Region("Invalid", rainfall)


def test_crop_rule_classifies_drought_good_and_waterlogging_risk():
    rule = CropRule("maize", 100, 200, "source")

    assert rule.classify_month(90) == "Drought risk for maize"
    assert rule.classify_month(150) == "Good for maize"
    assert rule.classify_month(210) == "Waterlogging risk for maize"
    assert rule.source == "source"


def test_cosine_similarity_matches_scipy_and_is_scale_invariant():
    first = np.array([120, 150, 200, 100], dtype=float)
    second = np.array([60, 75, 100, 50], dtype=float)

    value = cosine_similarity(first, second)

    assert value == pytest.approx(1 - scipy_cosine_distance(first, second))
    assert value == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("first", "second"),
    [([1, 2], [1]), ([0, 0], [1, 2]), ([np.nan, 2], [1, 2])],
)
def test_cosine_similarity_rejects_invalid_vectors(first, second):
    with pytest.raises(ValueError):
        cosine_similarity(first, second)


def test_similarity_matrices_have_expected_properties():
    regions = create_regions()
    matrices = pairwise_region_metrics(regions)

    for metric in matrices.values():
        np.testing.assert_allclose(metric.to_numpy(), metric.to_numpy().T)
    np.testing.assert_allclose(np.diag(matrices["cosine_similarity"]), 1)
    np.testing.assert_allclose(np.diag(matrices["pearson_correlation"]), 1)
    np.testing.assert_allclose(np.diag(matrices["euclidean_distance_mm"]), 0)
    assert matrices["cosine_similarity"].loc["Kampala", "Gulu"] > 0


def test_crop_classification_covers_every_region_month_and_crop():
    regions = create_regions()
    crops = create_crop_rules()
    classified = classify_months(regions, crops)

    assert len(classified) == len(regions) * 12 * len(crops)
    assert set(classified["classification"].str.split().str[0]) <= {
        "Good", "Drought", "Waterlogging"
    }
    assert all(crop.source for crop in crops)


def test_rainy_season_peak_detection_matches_expected_uganda_modes():
    regions = create_regions()
    seasons = analyze_rainy_seasons(regions).set_index("region")

    assert seasons.loc["Kampala", "detected_mode"] == "Bimodal"
    assert seasons.loc["Gulu", "detected_mode"] == "Unimodal"
    assert seasons.loc["Mbarara", "detected_mode"] == "Bimodal"
    assert seasons["matches_expected_mode"].all()
    assert detect_rainfall_peaks([5] * 12).size == 0


def test_region_summary_and_advisory_are_well_formed():
    regions = create_regions()
    summary = region_summary(regions)
    advisory = farmer_advisory(regions[1], create_crop_rules())

    assert summary.loc["Kampala", "annual_total_mm"] == pytest.approx(1600)
    assert "Gulu" in advisory and "maize" in advisory
    assert len(advisory.split()) <= 200


def test_run_analysis_creates_three_visual_outputs_or_summaries():
    results = run_analysis()

    assert len(results["classifications"]) == 108
    assert set(results["similarities"]) == {
        "cosine_similarity", "pearson_correlation", "euclidean_distance_mm"
    }
    assert results["plot_path"].is_file()


def test_nasa_power_fetch_converts_mean_daily_rain_to_monthly_total(
    tmp_path, monkeypatch
):
    region_name = next(iter(NASA_POWER_LOCATIONS))
    monthly_values = {
        f"{year}{month:02d}": float(month)
        for year in range(2023, 2033)
        for month in range(1, 13)
    }
    monthly_values["202313"] = 99.0
    monthly_values["203213"] = 99.0
    response_payload = {
        "properties": {
            "parameter": {
                "PRECTOTCORR": monthly_values
            }
        },
        "parameters": {"PRECTOTCORR": {"units": "mm/day"}},
        "header": {"fill_value": -999.0},
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response_payload).encode("utf-8")

    monkeypatch.setattr("src.rainfall_analysis.urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(
        "src.rainfall_analysis.NASA_POWER_LOCATIONS",
        {region_name: NASA_POWER_LOCATIONS[region_name]},
    )
    path = tmp_path / "rainfall.csv"

    data = download_nasa_power_rainfall(2023, 2032, path)

    assert len(data) == 12 * 10
    assert data.loc[(data["year"] == 2023) & (data["month_number"] == 1), "rainfall_mm"].iloc[0] == 31
    assert data.loc[(data["year"] == 2023) & (data["month_number"] == 2), "rainfall_mm"].iloc[0] == 56
    assert path.is_file()


def test_nasa_monthly_boxplot_requires_ten_years_not_needed_but_creates_plot(tmp_path):
    rows = [
        {"region": "Kampala", "year": year, "month_number": month, "rainfall_mm": month * year / 2020}
        for year in range(2015, 2025)
        for month in range(1, 13)
    ]
    output = plot_nasa_monthly_boxplots(pd.DataFrame(rows), tmp_path / "boxplots.png")

    assert output.is_file()