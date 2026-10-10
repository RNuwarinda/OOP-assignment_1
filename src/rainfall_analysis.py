import calendar
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine as scipy_cosine_distance

try:
    from .rainfall import CropRule, MONTHS, Region
except ImportError:
    from rainfall import CropRule, MONTHS, Region


ILLUSTRATIVE_RAINFALL_MM = {
    "Kampala": [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130],
    "Gulu": [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15],
    "Mbarara": [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90],
}

NASA_POWER_LOCATIONS = {
    "Kampala": {"latitude": 0.3476, "longitude": 32.5825},
    "Gulu": {"latitude": 2.7746, "longitude": 32.2990},
    "Mbarara": {"latitude": -0.6072, "longitude": 30.6545},
}

NASA_POWER_MONTHLY_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
NASA_POWER_REFERENCE = "https://power.larc.nasa.gov/docs/services/api/temporal/monthly/"
FAO_CROP_WATER_REFERENCE = "https://www.fao.org/4/s2022e/s2022e07.htm"
FAO_CROP_ET_REFERENCE = "https://www.fao.org/4/x0490e/x0490e0f.htm"
WORLD_BANK_UGANDA_CLIMATE_REFERENCE = (
    "https://climateknowledgeportal.worldbank.org/country/uganda/climate-data-historical"
)


def create_regions(rainfall_data=None):
    rainfall_data = ILLUSTRATIVE_RAINFALL_MM if rainfall_data is None else rainfall_data
    return [Region(name, values) for name, values in rainfall_data.items()]


def create_crop_rules():
    return [
        CropRule(
            "maize",
            83,
            192,
            "FAO Training Manual 3, Table 14: 500-800 mm over 125-180 days; "
            "monthly proxy divides seasonal bounds by crop-duration bounds. "
            + FAO_CROP_WATER_REFERENCE,
        ),
        CropRule(
            "beans",
            82,
            158,
            "FAO Training Manual 3, Table 14: 300-500 mm over the dry-bean "
            "duration range of 95-110 days; monthly proxy divides seasonal "
            "bounds by crop-duration bounds. "
            + FAO_CROP_WATER_REFERENCE,
        ),
        CropRule(
            "coffee",
            86,
            171,
            "Heuristic monthly ETc proxy: FAO Training Manual 3 gives coffee "
            "Kc=0.95; applying it to a 3-6 mm/day ETo screening range and "
            "30-day month gives about 86-171 mm/month. "
            + FAO_CROP_WATER_REFERENCE,
        ),
    ]


def classify_months(regions, crop_rules):
    rows = []
    for region in regions:
        for month, rainfall in zip(MONTHS, region.rainfall_mm):
            for crop_rule in crop_rules:
                rows.append({
                    "region": region.name,
                    "month": month,
                    "rainfall_mm": float(rainfall),
                    "crop": crop_rule.crop_name,
                    "classification": crop_rule.classify_month(rainfall),
                })
    return pd.DataFrame(rows)


def cosine_similarity(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("Cosine similarity requires equal-length one-dimensional vectors.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Cosine similarity vectors must be finite.")
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    similarity = float(np.dot(first, second) / denominator)
    scipy_similarity = 1.0 - float(scipy_cosine_distance(first, second))
    if not np.isclose(similarity, scipy_similarity, rtol=1e-12, atol=1e-12):
        raise ArithmeticError("NumPy and SciPy cosine similarity calculations disagree.")
    return similarity


def pairwise_region_metrics(regions):
    names = [region.name for region in regions]
    rainfall = np.stack([region.rainfall_mm for region in regions])
    cosine = np.eye(len(regions), dtype=float)
    pearson = np.corrcoef(rainfall)
    euclidean = np.zeros((len(regions), len(regions)), dtype=float)

    for first_index in range(len(regions)):
        for second_index in range(first_index + 1, len(regions)):
            cosine[first_index, second_index] = cosine_similarity(
                rainfall[first_index], rainfall[second_index]
            )
            cosine[second_index, first_index] = cosine[first_index, second_index]
            distance = float(np.linalg.norm(rainfall[first_index] - rainfall[second_index]))
            euclidean[first_index, second_index] = distance
            euclidean[second_index, first_index] = distance

    return {
        "cosine_similarity": pd.DataFrame(cosine, index=names, columns=names),
        "pearson_correlation": pd.DataFrame(pearson, index=names, columns=names),
        "euclidean_distance_mm": pd.DataFrame(euclidean, index=names, columns=names),
    }


def detect_rainfall_peaks(rainfall_mm, minimum_peak_separation=4, prominence_fraction=0.05):
    rainfall = np.asarray(rainfall_mm, dtype=float)
    if rainfall.shape != (12,) or not np.all(np.isfinite(rainfall)):
        raise ValueError("Peak detection requires 12 finite monthly rainfall values.")
    if np.any(rainfall < 0):
        raise ValueError("Rainfall values must be non-negative.")
    if not 1 <= minimum_peak_separation <= 12:
        raise ValueError("minimum_peak_separation must be between 1 and 12.")
    prominence = float(np.ptp(rainfall) * prominence_fraction)
    if np.ptp(rainfall) == 0:
        return np.asarray([], dtype=int)

    # Repeating the annual cycle lets peaks at the December/January boundary count.
    extended = np.tile(rainfall, 3)
    peaks, _ = find_peaks(
        extended,
        distance=minimum_peak_separation,
        prominence=prominence,
    )
    annual_peaks = peaks[(peaks >= 12) & (peaks < 24)] - 12
    return np.sort(annual_peaks.astype(int))


def analyze_rainy_seasons(regions, expected_modes=None):
    expected_modes = expected_modes or {
        "Kampala": "Bimodal",
        "Gulu": "Unimodal",
        "Mbarara": "Bimodal",
    }
    rows = []
    for region in regions:
        peaks = detect_rainfall_peaks(region.rainfall_mm)
        mode = "Bimodal" if len(peaks) >= 2 else "Unimodal"
        expected = expected_modes.get(region.name)
        rows.append({
            "region": region.name,
            "peak_months": [MONTHS[index] for index in peaks],
            "peak_count": len(peaks),
            "detected_mode": mode,
            "expected_uganda_zone_mode": expected,
            "matches_expected_mode": mode == expected if expected else None,
        })
    return pd.DataFrame(rows)


def region_summary(regions):
    return pd.DataFrame([
        {
            "region": region.name,
            "annual_total_mm": region.annual_total(),
            "monthly_mean_mm": region.mean_monthly_rainfall(),
            "wettest_month": region.wettest_month()[0],
            "wettest_mm": region.wettest_month()[1],
            "driest_month": region.driest_month()[0],
            "driest_mm": region.driest_month()[1],
            "rainfall_cv": region.coefficient_of_variation(),
        }
        for region in regions
    ]).set_index("region")


def _suitability_score(classification):
    if classification.startswith("Good"):
        return 0
    if classification.startswith("Drought"):
        return -1
    return 1


def plot_rainfall_and_suitability(regions, crop_rules, classifications, output_path):
    figure, (line_axis, heat_axis) = plt.subplots(
        2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1.2]}
    )
    colors = ("#245B8F", "#C75B39", "#2A9D8F")
    for region, color in zip(regions, colors):
        line_axis.plot(MONTHS, region.rainfall_mm, marker="o", color=color,
                       linewidth=2, label=region.name)
    line_axis.set_title("Illustrative monthly rainfall by region")
    line_axis.set_ylabel("Rainfall (mm)")
    line_axis.grid(axis="y", alpha=0.25)
    line_axis.legend(frameon=False, ncol=3)

    row_labels = []
    score_rows = []
    label_rows = []
    for region in regions:
        for rule in crop_rules:
            subset = classifications[
                (classifications["region"] == region.name)
                & (classifications["crop"] == rule.crop_name)
            ].set_index("month").loc[list(MONTHS)]
            scores = [_suitability_score(value) for value in subset["classification"]]
            labels = ["G" if score == 0 else "D" if score == -1 else "W" for score in scores]
            score_rows.append(scores)
            label_rows.append(labels)
            row_labels.append(f"{region.name} / {rule.crop_name}")

    color_map = ListedColormap(["#E7A79B", "#A8D5BA", "#E4C15C"])
    color_norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], color_map.N)
    heat_axis.imshow(np.asarray(score_rows), aspect="auto", cmap=color_map, norm=color_norm)
    heat_axis.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    heat_axis.set_xticks(np.arange(12), labels=MONTHS)
    heat_axis.set_title("Monthly crop-rainfall screen: G = good, D = drought, W = excess")
    heat_axis.set_xlabel("Month")
    for row_index, labels in enumerate(label_rows):
        for column_index, label in enumerate(labels):
            heat_axis.text(column_index, row_index, label, ha="center", va="center",
                           fontsize=8, color="#242424")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def farmer_advisory(region, crop_rules):
    maize_rule = next(rule for rule in crop_rules if rule.crop_name.lower() == "maize")
    good_months = [
        month for month, rainfall in zip(MONTHS, region.rainfall_mm)
        if maize_rule.classify_month(rainfall) == "Good for maize"
    ]
    first_good_month = good_months[0] if good_months else "the main rainy-season onset"
    return (
        f"Advisory for {region.name}: the illustrative rainfall screen places maize in "
        f"its monthly target range in {', '.join(good_months) if good_months else 'few months'}. "
        f"Where the first dependable rains begin, prepare land and plant near {first_good_month}; "
        "avoid sowing based on one isolated shower. Check local forecasts, soil moisture, "
        "seed maturity, drainage, and extension guidance before planting. This one-year "
        "illustrative profile is not a planting calendar or a guarantee of crop performance."
    )


def download_nasa_power_rainfall(start_year=2015, end_year=2024, output_path=None):
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise ValueError("start_year and end_year must be integers.")
    if end_year - start_year + 1 < 10:
        raise ValueError("Request at least 10 years of historical rainfall.")
    if start_year > end_year:
        raise ValueError("start_year must be no later than end_year.")

    rows = []
    for region_name, coordinates in NASA_POWER_LOCATIONS.items():
        query = urlencode({
            "parameters": "PRECTOTCORR",
            "community": "AG",
            "longitude": coordinates["longitude"],
            "latitude": coordinates["latitude"],
            "start": start_year,
            "end": end_year,
            "format": "JSON",
        })
        request = Request(
            f"{NASA_POWER_MONTHLY_URL}?{query}",
            headers={"User-Agent": "RainfallPatternCropSuitability/1.0"},
        )
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))

        rainfall_by_month = payload["properties"]["parameter"]["PRECTOTCORR"]
        units = payload["parameters"]["PRECTOTCORR"]["units"]
        if units != "mm/day":
            raise ValueError(f"Unexpected NASA POWER rainfall units: {units}")
        for key, mean_daily_rainfall in rainfall_by_month.items():
            year, month = int(key[:4]), int(key[4:])
            if not start_year <= year <= end_year or not 1 <= month <= 12:
                continue
            if mean_daily_rainfall == payload["header"]["fill_value"]:
                continue
            monthly_rainfall = float(mean_daily_rainfall) * calendar.monthrange(year, month)[1]
            rows.append({
                "region": region_name,
                "year": year,
                "month_number": month,
                "month": MONTHS[month - 1],
                "rainfall_mm": monthly_rainfall,
            })

    frame = pd.DataFrame(rows).sort_values(["region", "year", "month_number"])
    expected_rows = len(NASA_POWER_LOCATIONS) * (end_year - start_year + 1) * 12
    if len(frame) != expected_rows:
        raise ValueError(
            f"NASA POWER returned {len(frame)} valid month rows; expected {expected_rows}."
        )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
    return frame


def plot_nasa_monthly_boxplots(historical_rainfall, output_path):
    required = {"region", "year", "month_number", "rainfall_mm"}
    if not required.issubset(historical_rainfall.columns):
        raise ValueError(f"Historical rainfall must contain columns {sorted(required)}.")
    regions = list(dict.fromkeys(historical_rainfall["region"].tolist()))
    figure, axes = plt.subplots(len(regions), 1, figsize=(12, 3.4 * len(regions)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, region_name in zip(axes, regions):
        subset = historical_rainfall[historical_rainfall["region"] == region_name]
        month_data = [
            subset.loc[subset["month_number"] == month, "rainfall_mm"].to_numpy()
            for month in range(1, 13)
        ]
        axis.boxplot(month_data, tick_labels=MONTHS, showfliers=False)
        axis.set_ylabel("Rainfall (mm)")
        axis.set_title(region_name, loc="left")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Calendar month")
    figure.suptitle("NASA POWER monthly rainfall variability (10+ years)", y=1.002)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_real_data_extension(start_year=2015, end_year=2024):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / f"nasa_power_rainfall_{start_year}_{end_year}.csv"
    historical = download_nasa_power_rainfall(start_year, end_year, csv_path)
    plot_path = plot_nasa_monthly_boxplots(
        historical, project_root / "outputs" / "nasa_monthly_rainfall_boxplots.png"
    )
    return {"data": historical, "csv_path": csv_path, "plot_path": plot_path}


def run_analysis():
    regions = create_regions()
    crop_rules = create_crop_rules()
    classifications = classify_months(regions, crop_rules)
    similarities = pairwise_region_metrics(regions)
    seasons = analyze_rainy_seasons(regions)
    summaries = region_summary(regions)
    advisory_region = next(region for region in regions if region.name == "Gulu")
    project_root = Path(__file__).resolve().parent.parent
    plot_path = plot_rainfall_and_suitability(
        regions,
        crop_rules,
        classifications,
        project_root / "outputs" / "rainfall_crop_suitability.png",
    )
    return {
        "regions": regions,
        "crop_rules": crop_rules,
        "classifications": classifications,
        "similarities": similarities,
        "rainy_seasons": seasons,
        "region_summary": summaries,
        "advisory": farmer_advisory(advisory_region, crop_rules),
        "plot_path": plot_path,
        "expected_zone_reference": WORLD_BANK_UGANDA_CLIMATE_REFERENCE,
    }


def main():
    results = run_analysis()
    print("Illustrative rainfall region summaries:")
    print(results["region_summary"].round(2).to_string())
    for metric_name, matrix in results["similarities"].items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(matrix.round(3).to_string())
    print("\nRainy seasons detected (circular find_peaks):")
    print(results["rainy_seasons"].to_string(index=False))
    print("\nCrop monthly classifications:")
    print(results["classifications"].to_string(index=False))
    print("\nFarmer advisory:")
    print(results["advisory"])
    print(f"\nIllustrative rainfall plot saved to: {results['plot_path']}")


if __name__ == "__main__":
    main()