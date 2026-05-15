from dataclasses import dataclass


@dataclass(slots=True)
class AnalyticalShape:
    """
    Analytical shape inference for adaptive visualization.
    
    Tracks multiple dimensions of the data:
    - Temporal: time buckets (labels)
    - Categorical: dimensional groups (emirates, regions, categories)
    - Series: chart grouping axis (could be dimensions OR metrics)
    - Metrics: number of compared metrics in multi-metric queries
    """
    distinct_time_points: int
    distinct_groups: int
    distinct_series: int
    distinct_metrics: int
    row_count: int
    has_dimensions: bool
    has_metrics: bool

    @property
    def sparse_time_series(self) -> bool:
        """Only 1 time bucket but has temporal structure."""
        return self.distinct_time_points == 1 and self.has_dimensions

    @property
    def comparison_ready(self) -> bool:
        """Multiple categories available for comparison."""
        return self.distinct_groups >= 2

    @property
    def multi_metric_comparison(self) -> bool:
        """Comparing 2+ metrics."""
        return self.distinct_metrics >= 2

    @property
    def categorical_comparison(self) -> bool:
        """Multiple categories, no meaningful time axis."""
        return self.distinct_groups >= 2 and self.distinct_time_points <= 1

    @property
    def strong_trend(self) -> bool:
        """Enough time points for a real trend (3+)."""
        return self.distinct_time_points >= 3


def infer_shape(rows: list[dict]) -> AnalyticalShape:
    """
    Infer analytical shape from row structure.
    
    Examines multiple fields to understand data dimensionality:
    - label: primary x-axis (time bucket or category)
    - series: chart grouping (dimension or metric)
    - dimension: preserved Mongo dimensional grouping
    - metric: metric name in multi-metric comparisons
    """
    labels = set()
    series = set()
    dimensions = set()
    metrics = set()

    for row in rows:
        if label := row.get("label"):
            labels.add(str(label))
        
        if series_name := row.get("series"):
            series.add(str(series_name))
        
        if dimension := row.get("dimension"):
            dimensions.add(str(dimension))
        
        if metric := row.get("metric"):
            metrics.add(str(metric))

    return AnalyticalShape(
        # Temporal buckets (e.g., "2026-01", "2026-02")
        distinct_time_points=len(labels),
        
        # Real analytical categories (e.g., Fujairah, Ajman, Umm Al Quwain)
        # Prefers dimensions over series for accurate categorical counting
        distinct_groups=len(dimensions) if dimensions else len(series),
        
        # Chart series count (for multi-series rendering)
        # Could be metrics (in comparison) or dimensions (in grouped time)
        distinct_series=len(series) if series else 1,
        
        # Number of compared metrics (e.g., occupancy_rate_pct, revenues_collected_aed)
        distinct_metrics=len(metrics) if metrics else 1,
        
        row_count=len(rows),
        has_dimensions=bool(dimensions),
        has_metrics=bool(metrics),
    )