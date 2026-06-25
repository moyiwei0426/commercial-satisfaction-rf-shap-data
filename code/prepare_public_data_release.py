import csv
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsSpatialIndex,
    QgsVectorLayer,
)


WORKSPACE = Path(__file__).resolve().parent
RAW_DIR = WORKSPACE / "poi-date-7.29(1)" / "poi-date-7.29"
SHAP_DIR = WORKSPACE / "outputs" / "dominant_driver_maps"
RF_OUT = WORKSPACE / "outputs" / "random_rf_figures"
RELEASE_DIR = WORKSPACE / "outputs" / "public_data_release"
DATA_DIR = RELEASE_DIR / "data"
CODE_DIR = RELEASE_DIR / "code"
FIGURE_DATA_DIR = RELEASE_DIR / "outputs" / "figure_data"

CATEGORY_FILES = {
    "CAT": RAW_DIR / "CAT_random.csv",
    "ISS": RAW_DIR / "ISS_random.csv",
    "LEX": RAW_DIR / "LEX_random.csv",
    "RTL": RAW_DIR / "RTL_random.csv",
}

FEATURES = ["Bea", "Saf", "Act", "Wth", "Dpr", "Bro"]
FEATURE_LABELS = {
    "Bea": "Aesthetics",
    "Saf": "Safety",
    "Act": "Vibrancy",
    "Wth": "Wealth",
    "Dpr": "Depression",
    "Bro": "Boring",
}
FEATURE_DESCRIPTIONS = {
    "Bea": "Street-view aesthetics perception score",
    "Saf": "Street-view safety perception score",
    "Act": "Street-view vibrancy perception score",
    "Wth": "Street-view wealth perception score",
    "Dpr": "Street-view depression perception score",
    "Bro": "Street-view boring perception score",
}

RENDER_CRS = QgsCoordinateReferenceSystem("EPSG:32649")
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")
MAX_MATCH_DISTANCE_M = 200.0


def reset_release_dir():
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
    for directory in (DATA_DIR, CODE_DIR, FIGURE_DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def open_project():
    project = QgsProject.instance()
    candidates = sorted(WORKSPACE.glob("*修复.qgz")) + sorted(WORKSPACE.glob("*.qgz"))
    for candidate in candidates:
        if project.read(str(candidate)):
            print(f"Loaded QGIS project: {candidate.name}", flush=True)
            return project
    raise RuntimeError("Could not load a QGIS project.")


def find_layer(project, substring):
    for layer in project.mapLayers().values():
        if substring.lower() in layer.name().lower():
            return layer
    raise RuntimeError(f"Layer not found: {substring}")


def study_extent(reference_layer, project):
    transform = QgsCoordinateTransform(reference_layer.crs(), RENDER_CRS, project)
    extent = transform.transformBoundingBox(reference_layer.extent())
    extent.scale(1.055)
    return extent


def prepare_roads(source, extent, project):
    transform = QgsCoordinateTransform(source.crs(), RENDER_CRS, project)
    roads = []
    for source_feature in source.getFeatures():
        geometry = QgsGeometry(source_feature.geometry())
        if geometry.isEmpty():
            continue
        geometry.transform(transform)
        if not geometry.boundingBox().intersects(extent):
            continue
        roads.append((int(source_feature.id()), geometry))
    print(f"Prepared {len(roads):,} street segments within the study extent", flush=True)
    return roads


def build_road_layer(roads):
    layer = QgsVectorLayer("MultiLineString?crs=EPSG:32649&field=source_id:long", "public_roads", "memory")
    provider = layer.dataProvider()
    features = []
    from qgis.core import QgsFeature

    for source_id, geometry in roads:
        feature = QgsFeature(layer.fields())
        feature.setGeometry(geometry)
        feature["source_id"] = source_id
        features.append(feature)
    provider.addFeatures(features)
    layer.updateExtents()
    geometries = {}
    source_ids = {}
    index = QgsSpatialIndex()
    for feature in layer.getFeatures():
        index.addFeature(feature)
        geometries[feature.id()] = feature.geometry()
        source_ids[feature.id()] = int(feature["source_id"])
    return index, geometries, source_ids


def nearest_segment(point, index, geometries):
    candidates = index.nearestNeighbor(point, 8, MAX_MATCH_DISTANCE_M)
    if not candidates:
        return None
    point_geometry = QgsGeometry.fromPointXY(point)
    distances = [(fid, geometries[fid].distance(point_geometry)) for fid in candidates]
    fid, distance = min(distances, key=lambda item: item[1])
    return fid if distance <= MAX_MATCH_DISTANCE_M else None


def write_model_input():
    rows_written = 0
    output = DATA_DIR / "rf_model_input_anonymized.csv"
    fieldnames = ["category", "satisfaction_score"] + [FEATURE_LABELS[f] for f in FEATURES]
    with output.open("w", encoding="utf-8-sig", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for category, path in CATEGORY_FILES.items():
            with path.open("r", encoding="gb18030", newline="") as in_handle:
                for row in csv.DictReader(in_handle):
                    try:
                        clean = {
                            "category": category,
                            "satisfaction_score": float(row["star"]),
                        }
                        for feature in FEATURES:
                            clean[FEATURE_LABELS[feature]] = float(row[feature])
                    except (KeyError, TypeError, ValueError):
                        continue
                    writer.writerow(clean)
                    rows_written += 1
    print(f"Wrote model input rows: {rows_written:,}", flush=True)


def aggregate_segments(index, geometries, source_ids, project):
    transform = QgsCoordinateTransform(WGS84, RENDER_CRS, project)
    all_segment_ids = set()
    category_results = {}
    matching_rows = []

    for category in CATEGORY_FILES:
        path = SHAP_DIR / f"{category.lower()}_dominant_driver.csv"
        aggregates = defaultdict(
            lambda: {
                "n_poi": 0,
                "satisfaction": 0.0,
                "features": [0.0] * 6,
                "signed_shap": [0.0] * 6,
                "abs_shap": [0.0] * 6,
            }
        )
        total = 0
        matched = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                total += 1
                try:
                    point = transform.transform(QgsPointXY(float(row["lon_wgs84"]), float(row["lat_wgs84"])))
                    segment_fid = nearest_segment(point, index, geometries)
                    if segment_fid is None:
                        continue
                    feature_values = [float(row[feature]) for feature in FEATURES]
                    shap_values = [float(row[f"shap_{feature}"]) for feature in FEATURES]
                    satisfaction = float(row["star"])
                except (KeyError, TypeError, ValueError):
                    continue
                item = aggregates[segment_fid]
                item["n_poi"] += 1
                item["satisfaction"] += satisfaction
                for pos, value in enumerate(feature_values):
                    item["features"][pos] += value
                for pos, value in enumerate(shap_values):
                    item["signed_shap"][pos] += value
                    item["abs_shap"][pos] += abs(value)
                matched += 1
                all_segment_ids.add(segment_fid)

        rows = []
        for segment_fid, item in aggregates.items():
            n_poi = item["n_poi"]
            feature_means = [value / n_poi for value in item["features"]]
            signed_means = [value / n_poi for value in item["signed_shap"]]
            abs_means = [value / n_poi for value in item["abs_shap"]]
            dominant_idx = max(range(6), key=lambda pos: abs_means[pos])
            negative_idx = min(range(6), key=lambda pos: signed_means[pos])
            rows.append(
                {
                    "category": category,
                    "source_segment_id": source_ids[segment_fid],
                    "segment_fid": segment_fid,
                    "n_poi": n_poi,
                    "mean_satisfaction": item["satisfaction"] / n_poi,
                    "mean_features": feature_means,
                    "mean_signed_shap": signed_means,
                    "mean_abs_shap": abs_means,
                    "dominant_shap_driver": FEATURE_LABELS[FEATURES[dominant_idx]],
                    "negative_shap_driver": FEATURE_LABELS[FEATURES[negative_idx]] if signed_means[negative_idx] < 0 else "",
                    "negative_shap_value": signed_means[negative_idx],
                }
            )
        ordered = sorted(row["mean_satisfaction"] for row in rows)
        threshold = ordered[max(0, int((len(ordered) - 1) * 0.25))]
        for row in rows:
            row["low_satisfaction_flag"] = row["mean_satisfaction"] <= threshold
            row["low_satisfaction_threshold"] = threshold
        category_results[category] = rows
        matching_rows.append(
            {
                "category": category,
                "input_poi_records": total,
                "matched_poi_records": matched,
                "matched_street_segments": len(rows),
                "matching_distance_m": MAX_MATCH_DISTANCE_M,
                "low_satisfaction_threshold": threshold,
            }
        )
        print(
            f"{category}: matched {matched:,}/{total:,}; segments={len(rows):,}; "
            f"low threshold={threshold:.3f}",
            flush=True,
        )

    public_ids = {fid: f"S{pos:06d}" for pos, fid in enumerate(sorted(all_segment_ids), start=1)}
    return category_results, public_ids, matching_rows


def write_segment_tables(category_results, public_ids, geometries, matching_rows):
    analysis_fields = [
        "segment_id",
        "category",
        "n_poi",
        "mean_satisfaction",
        "mean_Safety",
        "mean_Aesthetics",
        "mean_Vibrancy",
        "mean_Wealth",
        "mean_Depression",
        "mean_Boring",
        "dominant_shap_driver",
        "negative_shap_driver",
        "negative_shap_value",
        "low_satisfaction_flag",
        "low_satisfaction_threshold",
        "geometry_wkt_epsg32649",
        "centroid_x_epsg32649",
        "centroid_y_epsg32649",
    ]
    shap_fields = [
        "segment_id",
        "category",
        "n_poi",
        "mean_shap_Safety",
        "mean_shap_Aesthetics",
        "mean_shap_Vibrancy",
        "mean_shap_Wealth",
        "mean_shap_Depression",
        "mean_shap_Boring",
        "mean_abs_shap_Safety",
        "mean_abs_shap_Aesthetics",
        "mean_abs_shap_Vibrancy",
        "mean_abs_shap_Wealth",
        "mean_abs_shap_Depression",
        "mean_abs_shap_Boring",
        "dominant_driver",
        "negative_driver",
        "negative_shap_value",
    ]

    analysis_path = DATA_DIR / "street_segment_analysis_data.csv"
    shap_path = DATA_DIR / "street_segment_shap_summary.csv"
    with analysis_path.open("w", encoding="utf-8-sig", newline="") as analysis_handle, shap_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as shap_handle:
        analysis_writer = csv.DictWriter(analysis_handle, fieldnames=analysis_fields)
        shap_writer = csv.DictWriter(shap_handle, fieldnames=shap_fields)
        analysis_writer.writeheader()
        shap_writer.writeheader()

        for category, rows in category_results.items():
            for row in rows:
                fid = row["segment_fid"]
                geometry = geometries[fid]
                centroid = geometry.centroid().asPoint()
                means = dict(zip((FEATURE_LABELS[f] for f in FEATURES), row["mean_features"]))
                signed = dict(zip((FEATURE_LABELS[f] for f in FEATURES), row["mean_signed_shap"]))
                abs_values = dict(zip((FEATURE_LABELS[f] for f in FEATURES), row["mean_abs_shap"]))
                public_segment_id = public_ids[fid]
                analysis_writer.writerow(
                    {
                        "segment_id": public_segment_id,
                        "category": category,
                        "n_poi": row["n_poi"],
                        "mean_satisfaction": row["mean_satisfaction"],
                        "mean_Safety": means["Safety"],
                        "mean_Aesthetics": means["Aesthetics"],
                        "mean_Vibrancy": means["Vibrancy"],
                        "mean_Wealth": means["Wealth"],
                        "mean_Depression": means["Depression"],
                        "mean_Boring": means["Boring"],
                        "dominant_shap_driver": row["dominant_shap_driver"],
                        "negative_shap_driver": row["negative_shap_driver"],
                        "negative_shap_value": row["negative_shap_value"],
                        "low_satisfaction_flag": int(row["low_satisfaction_flag"]),
                        "low_satisfaction_threshold": row["low_satisfaction_threshold"],
                        "geometry_wkt_epsg32649": geometry.asWkt(),
                        "centroid_x_epsg32649": centroid.x(),
                        "centroid_y_epsg32649": centroid.y(),
                    }
                )
                shap_writer.writerow(
                    {
                        "segment_id": public_segment_id,
                        "category": category,
                        "n_poi": row["n_poi"],
                        "mean_shap_Safety": signed["Safety"],
                        "mean_shap_Aesthetics": signed["Aesthetics"],
                        "mean_shap_Vibrancy": signed["Vibrancy"],
                        "mean_shap_Wealth": signed["Wealth"],
                        "mean_shap_Depression": signed["Depression"],
                        "mean_shap_Boring": signed["Boring"],
                        "mean_abs_shap_Safety": abs_values["Safety"],
                        "mean_abs_shap_Aesthetics": abs_values["Aesthetics"],
                        "mean_abs_shap_Vibrancy": abs_values["Vibrancy"],
                        "mean_abs_shap_Wealth": abs_values["Wealth"],
                        "mean_abs_shap_Depression": abs_values["Depression"],
                        "mean_abs_shap_Boring": abs_values["Boring"],
                        "dominant_driver": row["dominant_shap_driver"],
                        "negative_driver": row["negative_shap_driver"],
                        "negative_shap_value": row["negative_shap_value"],
                    }
                )

    with (DATA_DIR / "street_segment_matching_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matching_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matching_rows)


def write_variable_dictionary():
    rows = [
        ("category", "Commercial category: CAT, ISS, LEX, or RTL"),
        ("satisfaction_score", "Commercial satisfaction score used as the dependent variable"),
        ("mean_satisfaction", "Street-segment-level mean commercial satisfaction"),
        ("n_poi", "Number of matched commercial records aggregated to the street segment"),
        ("segment_id", "Anonymized street segment identifier assigned for this public release"),
        ("geometry_wkt_epsg32649", "Street segment geometry in WKT format, projected CRS EPSG:32649"),
        ("centroid_x_epsg32649", "Street segment centroid X coordinate in EPSG:32649"),
        ("centroid_y_epsg32649", "Street segment centroid Y coordinate in EPSG:32649"),
        ("low_satisfaction_flag", "1 if mean_satisfaction is in the bottom quartile within the category, otherwise 0"),
        ("low_satisfaction_threshold", "Category-specific bottom-quartile threshold for low satisfaction"),
        ("dominant_shap_driver", "Variable with the largest mean absolute SHAP value on the street segment"),
        ("negative_shap_driver", "Variable with the most negative mean signed SHAP value on the street segment"),
        ("negative_shap_value", "Mean signed SHAP value of the negative_shap_driver"),
    ]
    for feature in FEATURES:
        label = FEATURE_LABELS[feature]
        rows.extend(
            [
                (label, FEATURE_DESCRIPTIONS[feature]),
                (f"mean_{label}", f"Street-segment-level mean {label} perception score"),
                (f"mean_shap_{label}", f"Street-segment-level mean signed SHAP value for {label}"),
                (f"mean_abs_shap_{label}", f"Street-segment-level mean absolute SHAP value for {label}"),
            ]
        )
    with (DATA_DIR / "variable_dictionary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "description"])
        writer.writerows(rows)


def copy_supporting_outputs_and_code():
    for source_name in [
        "random_rf_metrics.csv",
        "random_rf_feature_importance.csv",
        "random_rf_params.json",
    ]:
        source = RF_OUT / source_name
        if source.exists():
            shutil.copy2(source, FIGURE_DATA_DIR / source_name)

    for script_name in [
        "prepare_public_data_release.py",
        "public_rf_training_from_release.py",
        "generate_random_rf_figures.py",
        "generate_random_rf_fig5.py",
        "generate_random_rf_fig6_fig7.py",
        "generate_random_rf_fig6_separate.py",
        "render_street_segment_layers_cli.py",
    ]:
        source = WORKSPACE / script_name
        if source.exists():
            shutil.copy2(source, CODE_DIR / script_name)


def write_readme():
    readme = """# Public Data Release

This folder contains the anonymized data and code required to reproduce the main random forest, SHAP, and street-segment-level spatial analyses.

## Released data

- `data/rf_model_input_anonymized.csv`: model input table containing only commercial category, satisfaction score, and six street-view perception variables. It does not include merchant names, exact addresses, phone numbers, platform IDs, review texts, or coordinates.
- `data/street_segment_analysis_data.csv`: street-segment-level aggregated commercial satisfaction, perception means, dominant SHAP driver, negative SHAP driver, low-satisfaction flag, and street-segment geometry.
- `data/street_segment_shap_summary.csv`: street-segment-level signed and absolute SHAP summaries for the six perception variables.
- `data/street_segment_matching_summary.csv`: summary of POI-to-street-segment matching counts and low-satisfaction thresholds.
- `data/variable_dictionary.csv`: field definitions.
- `outputs/figure_data/`: model metrics, feature importance, and random forest parameters used for the figures.

## Code

- `code/public_rf_training_from_release.py`: trains the random forest models from the released anonymized model-input table and reproduces model metrics and feature-importance summaries.
- `code/prepare_public_data_release.py`: author-side script used to generate this release package from non-public source files.
- Other scripts are author-side figure-generation scripts retained for transparency. Some of them require non-public raw inputs and are not expected to run from the released data alone.

To run the public random-forest reproduction script, install the dependencies listed in `requirements.txt`.

## Not released

The raw commercial POI and review records are not included. Merchant names, addresses, contact information, user or platform identifiers, review texts, raw URLs, and exact point-level commercial locations were removed from the public package. Point-level commercial POI analysis data are also excluded to reduce re-identification risk.

## Spatial aggregation

Commercial records were matched to OSM street segments within a maximum distance of 200 m. For each commercial category and street segment, satisfaction, perception variables, and SHAP values were averaged. Low satisfaction was defined as the bottom quartile of street-segment mean satisfaction within each commercial category. The dominant driver is the perception variable with the largest mean absolute SHAP value; the negative driver is the perception variable with the most negative mean signed SHAP value.

## Commercial categories

- CAT: catering
- ISS: in-store services
- LEX: leisure experience
- RTL: retail

## Perception variables

- Safety
- Aesthetics
- Vibrancy
- Wealth
- Depression
- Boring

## Suggested Data Availability Statement

The anonymized data and code supporting the findings of this study are available at [repository link]. The released dataset includes random forest model input data without location or merchant identifiers, street-segment-level commercial satisfaction indicators, aggregated street-view perception measures, and street-segment-level SHAP summary results. To protect privacy and comply with data-use restrictions, raw merchant names, addresses, contact information, review texts, platform identifiers, exact point-level commercial locations, and point-level commercial POI analysis data are not publicly released.
"""
    (RELEASE_DIR / "README.md").write_text(readme, encoding="utf-8")
    (RELEASE_DIR / "requirements.txt").write_text("pandas\nscikit-learn\n", encoding="utf-8")


def zip_release():
    zip_path = WORKSPACE / "outputs" / "public_data_release.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in RELEASE_DIR.rglob("*"):
            archive.write(path, path.relative_to(RELEASE_DIR.parent))
    print(f"Wrote {zip_path}", flush=True)


def main():
    reset_release_dir()
    write_model_input()

    qgs = QgsApplication([], False)
    QgsApplication.setPrefixPath("C:/Program Files/QGIS 3.40.9/apps/qgis-ltr", True)
    qgs.initQgis()
    try:
        project = open_project()
        sva_layer = find_layer(project, "3ring_SVA_EMO")
        osm_roads = find_layer(project, "路网合集")
        extent = study_extent(sva_layer, project)
        roads = prepare_roads(osm_roads, extent, project)
        index, geometries, source_ids = build_road_layer(roads)
        category_results, public_ids, matching_rows = aggregate_segments(index, geometries, source_ids, project)
        write_segment_tables(category_results, public_ids, geometries, matching_rows)
    finally:
        qgs.exitQgis()

    write_variable_dictionary()
    copy_supporting_outputs_and_code()
    write_readme()
    zip_release()


if __name__ == "__main__":
    main()
