import csv
from collections import defaultdict
from pathlib import Path

from qgis.PyQt.QtCore import QPointF, QRectF, QSize, Qt, QVariant
from qgis.PyQt.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPen
from qgis.core import (
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsGradientColorRamp,
    QgsGraduatedSymbolRenderer,
    QgsLineSymbol,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsPointXY,
    QgsProject,
    QgsRendererCategory,
    QgsSingleSymbolRenderer,
    QgsSpatialIndex,
    QgsVectorLayer,
)


WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "poi-date-7.29(1)" / "poi-date-7.29"
SHAP_DIR = WORKSPACE / "outputs" / "dominant_driver_maps"
PROJECT_PATH = WORKSPACE / "区位图_修复.qgz"
OUTPUT_DIR = WORKSPACE / "outputs" / "qgis" / "street_segment_layers"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
LEGEND_DIR = OUTPUT_DIR / "legends"
for directory in (OUTPUT_DIR, OVERLAY_DIR, LEGEND_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SIZE = 2400
RENDER_CRS = QgsCoordinateReferenceSystem("EPSG:32649")
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")
BACKGROUND = QColor("#061a33")
TRANSPARENT = QColor(0, 0, 0, 0)
FEATURES = ["Bea", "Saf", "Act", "Wth", "Dpr", "Bro"]
FEATURE_LABELS = {
    "Bea": "Aesthetics",
    "Saf": "Safety",
    "Act": "Vibrancy",
    "Wth": "Wealth",
    "Dpr": "Depression",
    "Bro": "Boring",
}
DRIVER_COLORS = {
    "Bea": "#4cc9f0",
    "Saf": "#51cf66",
    "Act": "#ffb000",
    "Wth": "#b197fc",
    "Dpr": "#ff5c5c",
    "Bro": "#c49a6c",
}
SATISFACTION_RAMPS = {
    "CAT": ("#542a16", "#ff9f43"),
    "ISS": ("#4b1f46", "#ff6fbd"),
    "LEX": ("#294520", "#a7ed62"),
    "RTL": ("#183a55", "#59c7ff"),
}


def find_layer(project, substring):
    for layer in project.mapLayers().values():
        if substring.lower() in layer.name().lower():
            return layer
    raise RuntimeError(f"Layer not found: {substring}")


def study_extent(reference, project):
    transform = QgsCoordinateTransform(reference.crs(), RENDER_CRS, project)
    extent = transform.transformBoundingBox(reference.extent())
    extent.scale(1.055)
    return extent


def prepare_roads(source, extent, project):
    roads = QgsVectorLayer("MultiLineString?crs=EPSG:32649&field=source_id:long", "study_roads", "memory")
    provider = roads.dataProvider()
    transform = QgsCoordinateTransform(source.crs(), RENDER_CRS, project)
    batch = []
    for source_feature in source.getFeatures():
        geometry = QgsGeometry(source_feature.geometry())
        if geometry.isEmpty():
            continue
        geometry.transform(transform)
        if not geometry.boundingBox().intersects(extent):
            continue
        feature = QgsFeature(roads.fields())
        feature.setGeometry(geometry)
        feature["source_id"] = int(source_feature.id())
        batch.append(feature)
    provider.addFeatures(batch)
    roads.updateExtents()
    print(f"Prepared {roads.featureCount():,} OSM street segments", flush=True)
    return roads


def build_road_index(roads):
    geometries = {}
    index = QgsSpatialIndex()
    for feature in roads.getFeatures():
        index.addFeature(feature)
        geometries[feature.id()] = feature.geometry()
    return index, geometries


def nearest_segment(point, index, geometries, max_distance=200.0):
    candidates = index.nearestNeighbor(point, 8, max_distance)
    if not candidates:
        return None, None
    point_geometry = QgsGeometry.fromPointXY(point)
    distances = [(fid, geometries[fid].distance(point_geometry)) for fid in candidates]
    fid, distance = min(distances, key=lambda item: item[1])
    if distance > max_distance:
        return None, distance
    return fid, distance


def aggregate_category(code, index, geometries, project):
    path = SHAP_DIR / f"{code.lower()}_dominant_driver.csv"
    transform = QgsCoordinateTransform(WGS84, RENDER_CRS, project)
    aggregates = defaultdict(lambda: {"count": 0, "star": 0.0, "abs": [0.0] * 6, "signed": [0.0] * 6})
    total = 0
    matched = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            try:
                point = transform.transform(QgsPointXY(float(row["lon_wgs84"]), float(row["lat_wgs84"])))
                segment_id, _ = nearest_segment(point, index, geometries)
                if segment_id is None:
                    continue
                values = [float(row[f"shap_{feature}"]) for feature in FEATURES]
                star = float(row["star"])
            except (KeyError, TypeError, ValueError):
                continue
            item = aggregates[segment_id]
            item["count"] += 1
            item["star"] += star
            for position, value in enumerate(values):
                item["abs"][position] += abs(value)
                item["signed"][position] += value
            matched += 1

    segment_rows = []
    for segment_id, item in aggregates.items():
        count = item["count"]
        abs_means = [value / count for value in item["abs"]]
        signed_means = [value / count for value in item["signed"]]
        dominant = FEATURES[max(range(6), key=lambda position: abs_means[position])]
        negative_position = min(range(6), key=lambda position: signed_means[position])
        negative = FEATURES[negative_position] if signed_means[negative_position] < 0 else ""
        segment_rows.append(
            {
                "segment_id": segment_id,
                "count": count,
                "satisfaction": item["star"] / count,
                "dominant": dominant,
                "negative": negative,
                "negative_shap": signed_means[negative_position],
            }
        )

    ordered = sorted(row["satisfaction"] for row in segment_rows)
    threshold = ordered[max(0, int((len(ordered) - 1) * 0.25))]
    for row in segment_rows:
        row["low_satisfaction"] = row["satisfaction"] <= threshold
    print(
        f"{code}: matched {matched:,}/{total:,} POIs to {len(segment_rows):,} segments; "
        f"low-satisfaction threshold={threshold:.3f}",
        flush=True,
    )
    return segment_rows


def make_segment_layer(name, rows, geometries, intervention_only=False):
    layer = QgsVectorLayer("MultiLineString?crs=EPSG:32649", name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes(
        [
            QgsField("segment_id", QVariant.LongLong),
            QgsField("poi_count", QVariant.Int),
            QgsField("sat_mean", QVariant.Double),
            QgsField("dominant", QVariant.String),
            QgsField("negative", QVariant.String),
            QgsField("neg_shap", QVariant.Double),
            QgsField("low_sat", QVariant.Bool),
        ]
    )
    layer.updateFields()
    features = []
    for row in rows:
        if intervention_only and not (row["low_satisfaction"] and row["negative"]):
            continue
        feature = QgsFeature(layer.fields())
        feature.setGeometry(geometries[row["segment_id"]])
        feature.setAttributes(
            [
                row["segment_id"],
                row["count"],
                row["satisfaction"],
                row["dominant"],
                row["negative"],
                row["negative_shap"],
                row["low_satisfaction"],
            ]
        )
        features.append(feature)
    provider.addFeatures(features)
    layer.updateExtents()
    return layer


def style_base_roads(layer):
    symbol = QgsLineSymbol.createSimple({"color": "255,255,255,115", "width": "0.10"})
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))


def style_satisfaction(layer, low, high):
    symbol = QgsLineSymbol.createSimple({"color": high, "width": "0.62", "capstyle": "round"})
    symbol.setOpacity(0.88)
    renderer = QgsGraduatedSymbolRenderer.createRenderer(
        layer,
        "sat_mean",
        5,
        QgsGraduatedSymbolRenderer.Quantile,
        symbol,
        QgsGradientColorRamp(QColor(low), QColor(high)),
    )
    for value_range in renderer.ranges():
        value_range.symbol().setOpacity(0.88)
    layer.setRenderer(renderer)


def style_categories(layer, field, width):
    categories = []
    for feature in FEATURES:
        symbol = QgsLineSymbol.createSimple(
            {"color": DRIVER_COLORS[feature], "width": str(width), "capstyle": "round"}
        )
        symbol.setOpacity(0.92)
        categories.append(QgsRendererCategory(feature, symbol, FEATURE_LABELS[feature]))
    layer.setRenderer(QgsCategorizedSymbolRenderer(field, categories))


def render(layer, extent, output, background):
    settings = QgsMapSettings()
    settings.setOutputSize(QSize(SIZE, SIZE))
    settings.setOutputDpi(300)
    settings.setBackgroundColor(background)
    settings.setDestinationCrs(RENDER_CRS)
    settings.setExtent(extent)
    settings.setLayers([layer])
    settings.setFlag(QgsMapSettings.Antialiasing, True)
    job = QgsMapRendererParallelJob(settings)
    job.start()
    job.waitForFinished()
    image = job.renderedImage()
    if not image.save(str(output), "PNG"):
        raise RuntimeError(f"Could not save {output}")
    print(f"Saved: {output}", flush=True)


def transparent_image(width, height):
    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    return image


def draw_text(painter, x, y, text, size=16):
    painter.setPen(QColor("#ffffff"))
    painter.setFont(QFont("Arial", size))
    painter.drawText(QPointF(x, y), text)


def export_legends():
    for code, (low, high) in SATISFACTION_RAMPS.items():
        image = transparent_image(580, 105)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        gradient = QLinearGradient(30, 35, 550, 35)
        gradient.setColorAt(0, QColor(low))
        gradient.setColorAt(1, QColor(high))
        painter.setPen(QPen(QColor(255, 255, 255, 150), 2))
        painter.setBrush(gradient)
        painter.drawRoundedRect(QRectF(30, 20, 520, 32), 5, 5)
        draw_text(painter, 30, 88, "Low")
        draw_text(painter, 500, 88, "High")
        painter.end()
        image.save(str(LEGEND_DIR / f"legend_{code.lower()}_satisfaction.png"), "PNG")

    image = transparent_image(360, 330)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    for index, feature in enumerate(FEATURES):
        y = 35 + index * 48
        painter.setPen(QPen(QColor(DRIVER_COLORS[feature]), 14))
        painter.drawLine(QPointF(25, y), QPointF(85, y))
        draw_text(painter, 105, y + 7, FEATURE_LABELS[feature], 15)
    painter.end()
    image.save(str(LEGEND_DIR / "legend_emotional_drivers.png"), "PNG")


def main():
    app = QgsApplication([], False)
    app.initQgis()
    try:
        project = QgsProject.instance()
        if not project.read(str(PROJECT_PATH)):
            raise RuntimeError(f"Could not read {PROJECT_PATH}")
        reference = find_layer(project, "3ring_SVA_EMO")
        road_source = find_layer(project, "武汉市_路网合集")
        extent = study_extent(reference, project)
        roads = prepare_roads(road_source, extent, project)
        index, geometries = build_road_index(roads)

        style_base_roads(roads)
        render(roads, extent, OUTPUT_DIR / "base_map.png", BACKGROUND)

        for code, (low, high) in SATISFACTION_RAMPS.items():
            rows = aggregate_category(code, index, geometries, project)

            satisfaction = make_segment_layer(f"{code}_satisfaction", rows, geometries)
            style_satisfaction(satisfaction, low, high)
            render(satisfaction, extent, OVERLAY_DIR / f"{code.lower()}_satisfaction.png", TRANSPARENT)

            dominant = make_segment_layer(f"{code}_dominant", rows, geometries)
            style_categories(dominant, "dominant", 0.72)
            render(dominant, extent, OVERLAY_DIR / f"{code.lower()}_dominant_driver.png", TRANSPARENT)

            intervention = make_segment_layer(f"{code}_intervention", rows, geometries, intervention_only=True)
            style_categories(intervention, "negative", 1.00)
            render(
                intervention,
                extent,
                OVERLAY_DIR / f"{code.lower()}_low_satisfaction_negative_driver.png",
                TRANSPARENT,
            )

        export_legends()
    finally:
        app.exitQgis()


if __name__ == "__main__":
    main()
