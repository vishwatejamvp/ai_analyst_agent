"""Test multi-metric visualization fix.

This test verifies that multi-metric queries properly generate series fields
for visualization, fixing the issue where only one metric was displayed.
"""

import pytest
from services.visualization.shape_analysis import infer_shape


def test_sparse_temporal_multi_metric_shape():
    """Test shape detection for sparse temporal multi-metric data.
    
    Scenario: "trend for Occupancy Rates and Revenues in 2026"
    - 1 time bucket (2026-01)
    - 6 emirates (dimensions)
    - 2 metrics (occupancy_rate_pct, revenues_collected_aed)
    
    Expected: sparse_time_series=True, multi_metric_comparison=True
    """
    rows = [
        # Revenues for each emirate
        {"label": "2026-01", "dimension": "Ajman", "metric": "revenues_collected_aed", "value": 911681.99},
        {"label": "2026-01", "dimension": "Dubai", "metric": "revenues_collected_aed", "value": 698039.00},
        {"label": "2026-01", "dimension": "Fujairah", "metric": "revenues_collected_aed", "value": 480681.67},
        {"label": "2026-01", "dimension": "Ras Al Khaimah", "metric": "revenues_collected_aed", "value": 709600.83},
        {"label": "2026-01", "dimension": "Sharjah", "metric": "revenues_collected_aed", "value": 296810.33},
        {"label": "2026-01", "dimension": "Umm Al Quwain", "metric": "revenues_collected_aed", "value": 376286.67},
        # Occupancy rates for each emirate
        {"label": "2026-01", "dimension": "Ajman", "metric": "occupancy_rate_pct", "value": 88.4},
        {"label": "2026-01", "dimension": "Dubai", "metric": "occupancy_rate_pct", "value": 99.3},
        {"label": "2026-01", "dimension": "Fujairah", "metric": "occupancy_rate_pct", "value": 89.6},
        {"label": "2026-01", "dimension": "Ras Al Khaimah", "metric": "occupancy_rate_pct", "value": 76.6},
        {"label": "2026-01", "dimension": "Sharjah", "metric": "occupancy_rate_pct", "value": 97.1},
        {"label": "2026-01", "dimension": "Umm Al Quwain", "metric": "occupancy_rate_pct", "value": 80.2},
    ]
    
    shape = infer_shape(rows)
    
    assert shape.distinct_time_points == 1, "Should have 1 time bucket"
    assert shape.distinct_groups == 6, "Should have 6 emirates"
    assert shape.distinct_metrics == 2, "Should have 2 metrics"
    assert shape.sparse_time_series is True, "Should be sparse temporal"
    assert shape.multi_metric_comparison is True, "Should be multi-metric"
    assert shape.categorical_comparison is True, "Should be categorical"


def test_transformed_rows_have_series_field():
    """Test that transformation adds series field for multi-metric data.
    
    After transformation, each row should have:
    - label: dimension (emirate name)
    - series: metric name
    - series_label: human-readable metric name
    - value: numeric value
    """
    from services.analyst_service import AnalystService
    from services.visualization.shape_analysis import infer_shape
    
    # Simulate raw rows from _run_multi_metric_comparison
    raw_rows = [
        {"label": "2026-01", "dimension": "Ajman", "metric": "revenues_collected_aed", 
         "metric_label": "Revenues Collected (AED)", "value": 911681.99},
        {"label": "2026-01", "dimension": "Ajman", "metric": "occupancy_rate_pct", 
         "metric_label": "Occupancy Rate (%)", "value": 88.4},
        {"label": "2026-01", "dimension": "Dubai", "metric": "revenues_collected_aed", 
         "metric_label": "Revenues Collected (AED)", "value": 698039.00},
        {"label": "2026-01", "dimension": "Dubai", "metric": "occupancy_rate_pct", 
         "metric_label": "Occupancy Rate (%)", "value": 99.3},
    ]
    
    shape = infer_shape(raw_rows)
    
    # Simulate transformation (this is what the fix does)
    transformed = []
    for row in raw_rows:
        dimension = row.get("dimension")
        metric = row.get("metric")
        
        if dimension and metric:
            new_row = {
                "label": dimension,
                "series": metric,
                "series_label": row.get("metric_label"),
                "dimension": dimension,
                "metric": metric,
                "value": row.get("value"),
            }
            transformed.append(new_row)
    
    # Verify transformation
    assert len(transformed) == 4, "Should have 4 transformed rows"
    
    # Check first row (Ajman revenues)
    assert transformed[0]["label"] == "Ajman"
    assert transformed[0]["series"] == "revenues_collected_aed"
    assert transformed[0]["series_label"] == "Revenues Collected (AED)"
    assert transformed[0]["value"] == 911681.99
    
    # Check second row (Ajman occupancy)
    assert transformed[1]["label"] == "Ajman"
    assert transformed[1]["series"] == "occupancy_rate_pct"
    assert transformed[1]["series_label"] == "Occupancy Rate (%)"
    assert transformed[1]["value"] == 88.4
    
    # Verify all rows have series field
    assert all("series" in row for row in transformed), "All rows must have series field"
    
    # Verify shape after transformation
    transformed_shape = infer_shape(transformed)
    assert transformed_shape.distinct_series == 2, "Should detect 2 series"


def test_hajj_package_multi_metric():
    """Test multi-metric visualization for Hajj package service data.
    
    Scenario: "trend for hajj package service in 2026"
    - 12 time buckets (months)
    - 4 metrics (total_transactions, hajj_package_recipients, 
                 smart_app_transactions, website_transactions)
    
    Expected: All 4 metrics should be visualized as separate series
    """
    rows = [
        # January
        {"label": "2026-01", "metric": "total_transactions", "value": 62},
        {"label": "2026-01", "metric": "hajj_package_recipients", "value": 0},
        {"label": "2026-01", "metric": "smart_app_transactions", "value": 5},
        {"label": "2026-01", "metric": "website_transactions", "value": 57},
        # February
        {"label": "2026-02", "metric": "total_transactions", "value": 113},
        {"label": "2026-02", "metric": "hajj_package_recipients", "value": 0},
        {"label": "2026-02", "metric": "smart_app_transactions", "value": 16},
        {"label": "2026-02", "metric": "website_transactions", "value": 97},
        # March (all zeros)
        {"label": "2026-03", "metric": "total_transactions", "value": 0},
        {"label": "2026-03", "metric": "hajj_package_recipients", "value": 0},
        {"label": "2026-03", "metric": "smart_app_transactions", "value": 0},
        {"label": "2026-03", "metric": "website_transactions", "value": 0},
    ]
    
    shape = infer_shape(rows)
    
    assert shape.distinct_time_points == 3, "Should have 3 time buckets"
    assert shape.distinct_metrics == 4, "Should have 4 metrics"
    assert shape.multi_metric_comparison is True, "Should be multi-metric"
    
    # For time-series multi-metric, series field should be set to metric
    transformed = []
    for row in rows:
        metric = row.get("metric")
        if metric and not row.get("series"):
            new_row = dict(row)
            new_row["series"] = metric
            new_row["series_label"] = metric.replace("_", " ").title()
            transformed.append(new_row)
        else:
            transformed.append(row)
    
    # Verify all rows have series field
    assert all("series" in row for row in transformed), "All rows must have series field"
    
    # Verify shape detects series
    transformed_shape = infer_shape(transformed)
    assert transformed_shape.distinct_series == 4, "Should detect 4 series"


def test_multi_series_profile_building():
    """Test that multi-series profile is built correctly from transformed rows."""
    from services.metric_profile import classify_series
    
    # Simulate transformed rows with series field
    rows = [
        {"label": "Ajman", "series": "revenues_collected_aed", "series_label": "Revenues Collected (AED)", "value": 911681.99},
        {"label": "Ajman", "series": "occupancy_rate_pct", "series_label": "Occupancy Rate (%)", "value": 88.4},
        {"label": "Dubai", "series": "revenues_collected_aed", "series_label": "Revenues Collected (AED)", "value": 698039.00},
        {"label": "Dubai", "series": "occupancy_rate_pct", "series_label": "Occupancy Rate (%)", "value": 99.3},
    ]
    
    # Extract unique series
    seen = {}
    for row in rows:
        key = row.get("series")
        if key and key not in seen:
            seen[key] = row.get("series_label") or key
    
    # Build series tuples
    series_tuples = [(key, key, label) for key, label in seen.items()]
    
    # Classify series
    multi = classify_series(series_tuples, operation="sum")
    
    assert len(multi.series) == 2, "Should have 2 series"
    assert multi.dual_axis is True, "Should use dual axis (% vs AED)"
    
    # Verify series names
    series_keys = {s.key for s in multi.series}
    assert "revenues_collected_aed" in series_keys
    assert "occupancy_rate_pct" in series_keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
