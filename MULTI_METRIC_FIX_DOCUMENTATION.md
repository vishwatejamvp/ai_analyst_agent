# Multi-Metric Visualization Fix Documentation

## Problem Summary

The system was failing to visualize multiple metrics in queries like:
1. **"trend for Occupancy Rates and Revenues in 2026"** - Only showing revenues, missing occupancy rates
2. **"trend for hajj package service in 2026"** - Only showing total_transactions, missing the other 3 metrics

## Root Cause

The multi-metric comparison pipeline was adding `metric` and `dimension` fields to rows but **NOT setting the `series` field**, which is required for multi-series chart rendering.

### The Detection Chain

```python
# Line 3997 in analyst_service.py
if any("series" in row for row in transformed_rows):
    multi = self._build_multi_series_profile(transformed_rows, decision)
    # ... render multi-series chart
else:
    # ... render single-series chart (WRONG PATH for multi-metric!)
```

When the `series` field was missing, the system fell back to single-metric rendering, silently dropping all other metrics.

## The Fix

### 1. Updated `_run_multi_metric_comparison` (Lines 2589-2629)

**Before:**
```python
row["metric"] = metric
row["metric_label"] = self._human_label(metric).title()
# series field NOT set - causes detection failure
```

**After:**
```python
row["metric"] = metric
row["metric_label"] = self._human_label(metric).title()
# Note: series field will be set by transformation layer
# based on analytical shape (sparse temporal vs. strong trend)
```

### 2. Enhanced `_transform_rows_for_visualization` (Lines 3802-3865)

#### Sparse Temporal Multi-Metric (e.g., Occupancy + Revenues in 2026)

**Input:**
```python
[
  {"label": "2026-01", "dimension": "Ajman", "metric": "revenues_collected_aed", "value": 911681.99},
  {"label": "2026-01", "dimension": "Ajman", "metric": "occupancy_rate_pct", "value": 88.4},
  {"label": "2026-01", "dimension": "Dubai", "metric": "revenues_collected_aed", "value": 698039.00},
  {"label": "2026-01", "dimension": "Dubai", "metric": "occupancy_rate_pct", "value": 99.3},
  # ... more emirates
]
```

**Transformation Logic:**
```python
if shape.sparse_time_series and shape.categorical_comparison:
    for row in rows:
        dimension = row.get("dimension")
        metric = row.get("metric")
        
        new_row = {
            "label": dimension,              # Emirate becomes x-axis
            "series": metric,                # Metric becomes series grouping
            "series_label": metric_label,    # Human-readable name
            "value": row.get("value"),
        }
```

**Output:**
```python
[
  {"label": "Ajman", "series": "revenues_collected_aed", "series_label": "Revenues Collected (AED)", "value": 911681.99},
  {"label": "Ajman", "series": "occupancy_rate_pct", "series_label": "Occupancy Rate (%)", "value": 88.4},
  {"label": "Dubai", "series": "revenues_collected_aed", "series_label": "Revenues Collected (AED)", "value": 698039.00},
  {"label": "Dubai", "series": "occupancy_rate_pct", "series_label": "Occupancy Rate (%)", "value": 99.3},
  # ... more emirates
]
```

**Result:** Grouped bar chart with emirates on x-axis, two bars per emirate (revenues + occupancy)

#### Multi-Metric Categorical (e.g., Hajj Package Service)

**Input:**
```python
[
  {"label": "2026-01", "metric": "total_transactions", "value": 62},
  {"label": "2026-01", "metric": "smart_app_transactions", "value": 5},
  {"label": "2026-01", "metric": "website_transactions", "value": 57},
  {"label": "2026-01", "metric": "hajj_package_recipients", "value": 0},
  # ... more months
]
```

**Transformation Logic:**
```python
if shape.multi_metric_comparison and shape.categorical_comparison:
    for row in rows:
        metric = row.get("metric")
        if metric and not row.get("series"):
            new_row = dict(row)
            new_row["series"] = metric
            new_row["series_label"] = metric.replace("_", " ").title()
```

**Output:**
```python
[
  {"label": "2026-01", "series": "total_transactions", "series_label": "Total Transactions", "value": 62},
  {"label": "2026-01", "series": "smart_app_transactions", "series_label": "Smart App Transactions", "value": 5},
  {"label": "2026-01", "series": "website_transactions", "series_label": "Website Transactions", "value": 57},
  {"label": "2026-01", "series": "hajj_package_recipients", "series_label": "Hajj Package Recipients", "value": 0},
  # ... more months
]
```

**Result:** Multi-line chart with 4 series, one for each metric

### 3. Added Debug Logging (Lines 3990-4010)

```python
has_series = any("series" in row for row in transformed_rows)
logger.info(
    f"Multi-series detection: has_series={has_series}, "
    f"row_count={len(transformed_rows)}, "
    f"sample_row={transformed_rows[0] if transformed_rows else None}"
)

if has_series:
    multi = self._build_multi_series_profile(transformed_rows, decision)
    logger.info(
        f"Built multi-series profile: {len(multi.series)} series, "
        f"dual_axis={multi.dual_axis}, "
        f"series_names={[s.label for s in multi.series]}"
    )
```

This helps diagnose issues by showing:
- Whether series field is present
- How many series were detected
- Whether dual-axis is enabled (for mixed units like % + AED)

## How It Works

### Data Flow

```
1. User Query: "trend for Occupancy Rates and Revenues in 2026"
   ↓
2. _run_multi_metric_comparison detects 2 metrics
   ↓
3. Runs separate aggregations for each metric
   ↓
4. Adds metric + dimension fields to rows
   ↓
5. infer_shape analyzes row structure
   → sparse_time_series=True (1 time bucket)
   → multi_metric_comparison=True (2 metrics)
   → categorical_comparison=True (6 emirates)
   ↓
6. _transform_rows_for_visualization pivots data
   → dimension → label (x-axis)
   → metric → series (grouping)
   ↓
7. Multi-series detection succeeds (series field present)
   ↓
8. _build_multi_series_profile creates profile
   → 2 series: revenues_collected_aed, occupancy_rate_pct
   → dual_axis=True (AED vs %)
   ↓
9. Chart renders with grouped bars
   → X-axis: Emirates
   → Y-axis (left): Revenues (AED)
   → Y-axis (right): Occupancy (%)
   → 2 bars per emirate
```

## Testing

### Test Case 1: Occupancy + Revenues

**Query:** "trend for Occupancy Rates and Revenues in 2026"

**Expected Result:**
- Chart type: Grouped bar chart
- X-axis: 6 emirates (Ajman, Dubai, Fujairah, Ras Al Khaimah, Sharjah, Umm Al Quwain)
- Series: 2 (Revenues Collected AED, Occupancy Rate %)
- Dual-axis: Yes (AED on left, % on right)
- Bars per emirate: 2

### Test Case 2: Hajj Package Service

**Query:** "trend for hajj package service in 2026"

**Expected Result:**
- Chart type: Multi-line chart
- X-axis: 12 months (2026-01 to 2026-12)
- Series: 4 (Total Transactions, Hajj Package Recipients, Smart App Transactions, Website Transactions)
- Lines: 4 (one per metric)

## Verification Steps

1. **Check logs for multi-series detection:**
   ```
   Multi-series detection: has_series=True, row_count=12, sample_row={...}
   Built multi-series profile: 2 series, dual_axis=True, series_names=['Revenues Collected (AED)', 'Occupancy Rate (%)']
   ```

2. **Verify chart payload has series data:**
   - `series_count` should be > 1
   - `echarts_option` or `plotly_json` should have multiple series

3. **Check UI:**
   - Metric dropdown should show all metrics
   - Switching metrics should work
   - All metrics should be visible on the chart

## Files Modified

1. **services/analyst_service.py**
   - Lines 2589-2629: Updated `_run_multi_metric_comparison` comments
   - Lines 3802-3865: Enhanced `_transform_rows_for_visualization`
   - Lines 3990-4010: Added debug logging for multi-series detection

## Related Files

- **services/visualization/shape_analysis.py**: Detects analytical shape
- **services/visualization/visualization_planner.py**: Chooses chart type
- **services/metric_profile.py**: Builds multi-series profiles
- **services/chart_service.py**: Renders multi-series charts
- **services/chart_echarts.py**: ECharts multi-series rendering

## Key Insights

1. **The `series` field is critical** - Without it, multi-series detection fails
2. **Transformation is shape-aware** - Different transformations for sparse temporal vs. strong trends
3. **Dual-axis is automatic** - Mixed units (% + AED) trigger dual y-axis
4. **Logging is essential** - Debug logs help diagnose transformation issues

## Future Improvements

1. Add unit tests for transformation logic
2. Add integration tests for end-to-end multi-metric queries
3. Consider caching multi-metric aggregations
4. Add user-facing error messages when metrics are missing
5. Support more than 2 metrics with different units (currently limited to dual-axis)

## Troubleshooting

### Issue: Still only showing one metric

**Check:**
1. Are logs showing `has_series=True`?
2. Is `distinct_metrics` > 1 in shape analysis?
3. Are all metrics present in the raw data?

**Debug:**
```python
# Add to analyst_service.py before transformation
logger.info(f"Raw rows before transformation: {rows[:5]}")
logger.info(f"Shape: {shape}")
```

### Issue: Chart shows wrong axis labels

**Check:**
1. Is `multi_series_profile` being passed to chart renderer?
2. Are metric profiles correctly classified?
3. Is dual-axis enabled when needed?

**Debug:**
```python
# Check multi-series profile
logger.info(f"Multi profile: {multi}")
logger.info(f"Primary axis: {multi.primary_axis_label}")
logger.info(f"Secondary axis: {multi.secondary_axis_label}")
```

## Summary

This fix ensures that multi-metric queries properly generate the `series` field required for multi-series visualization. The transformation layer now intelligently pivots data based on analytical shape, enabling correct visualization of:

1. Sparse temporal multi-metric comparisons (dimension → x-axis, metric → series)
2. Time-series multi-metric comparisons (time → x-axis, metric → series)
3. Categorical multi-metric comparisons (category → x-axis, metric → series)

All metrics are now properly visualized with appropriate chart types, dual-axis support, and interactive metric selection.
