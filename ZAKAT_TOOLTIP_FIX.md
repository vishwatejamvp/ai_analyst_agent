# Zakat Payment Service Tooltip & Dropdown Fix

## Issue Description

When querying "Zakat Payment Service Statistics 2025", the system returns 19 payment channels (dimensions) but the tooltip and dropdown show "Funds Collected Aed" for all of them instead of the channel names.

### Current Behavior
- **Tooltip**: Shows "Funds Collected Aed" for all 19 lines
- **Dropdown**: Shows "All metrics" but should show channel names
- **Data**: Correctly has channel names in `dimension` and `series` fields

### Expected Behavior
- **Tooltip**: Should show channel name (e.g., "BANK-MOB: 35,864,605.40")
- **Dropdown**: Should show channel names (e.g., "BANK-MOB", "Direct Payers at the Authority")

## Root Cause

The data structure shows:
```
dimension: "BANK-MOB"
metric: "funds_collected_aed"
series: "BANK-MOB"
```

The tooltip is using `metric_label` instead of `series` name, causing all tooltips to show the same metric name.

## Solution

The tooltip formatter in ECharts needs to use the series name (channel name) instead of the metric name when displaying dimensional data.

### Files to Modify

1. **services/chart_echarts.py** - Update tooltip formatter
2. **static/index.html** - Ensure dropdown uses series names

## Implementation Status

- [ ] Update tooltip formatter to show series names
- [ ] Update dropdown to show channel names instead of metric names
- [ ] Test with Zakat Payment Service query
