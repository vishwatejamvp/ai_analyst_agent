from dataclasses import dataclass


@dataclass(slots=True)
class MetricSemantic:
    field: str
    semantic_type: str
    default_aggregation: str
    formatter: str


METRIC_REGISTRY: dict[str, MetricSemantic] = {
    "occupancy_rate_pct": MetricSemantic(
        field="occupancy_rate_pct",
        semantic_type="percentage",
        default_aggregation="avg",
        formatter="percent",
    ),

    "revenues_collected_aed": MetricSemantic(
        field="revenues_collected_aed",
        semantic_type="currency",
        default_aggregation="sum",
        formatter="aed",
    ),
}


def metric_semantic(metric: str | None) -> MetricSemantic | None:
    if not metric:
        return None
    return METRIC_REGISTRY.get(metric.lower())