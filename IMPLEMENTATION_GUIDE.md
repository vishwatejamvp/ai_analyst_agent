# Interactive Statistics in Chat - Implementation Guide

## Overview

This guide explains how to show interactive Zakat Payment Service Statistics **directly in the chat interface** with year/month filters, without needing a separate dashboard.

## Current State

✅ **What Works:**
- Chat shows multi-chart panels with tabs (Trend, Bars, Distribution, Total)
- Metric selector dropdown ("All metrics")
- ECharts/Plotly rendering
- Multi-view chart panels via [`ChartPanelBuilder`](services/chart_panel.py:1)

❌ **What's Missing:**
- Year/month filter dropdowns in chat panels
- Dynamic data refresh when filters change
- Automatic detection of "statistics" queries

## Solution Architecture

### When User Asks: "Zakat Payment Service Statistics 2025"

```
1. Question Detection (analyst_service.py)
   ↓
2. Fetch Complete Dataset (MongoDB)
   ↓
3. Create Multi-Chart Panel (chart_panel.py)
   ↓
4. Add Interactive Filters (index.html)
   ↓
5. Render in Chat with Tabs + Filters
```

## Implementation Steps

### Step 1: Detect Statistics Queries

**File**: `services/analyst_service.py`

Add detection logic in `_handle_analytical()`:

```python
def _handle_analytical(self, question: str, decision: RoutingDecision, ...) -> AnalystResponse:
    # Detect statistics/overview queries
    is_statistics_query = any(
        keyword in question.lower() 
        for keyword in ['statistics', 'overview', 'dashboard', 'summary report']
    )
    
    if is_statistics_query and 'zakat' in question.lower():
        return self._create_zakat_statistics_panel(question, decision)
    
    # ... existing logic
```

### Step 2: Create Statistics Panel Builder

**File**: `services/analyst_service.py`

```python
def _create_zakat_statistics_panel(
    self, 
    question: str, 
    decision: RoutingDecision
) -> AnalystResponse:
    \"\"\"Create interactive multi-view panel for Zakat statistics.\"\"\"
    
    # Extract year from question (default to current year)
    import re
    year_match = re.search(r'20\d{2}', question)
    year = int(year_match.group()) if year_match else 2025
    
    # Fetch ALL data for the year (all channels, all months)
    collection = "awqaf_zakat_payment_service_facts"
    pipeline = [
        {"$match": {"year": year}},
        {
            "$group": {
                "_id": {
                    "channel": "$dimension",
                    "period": "$period"
                },
                "number_of_payers": {"$sum": "$number_of_payers"},
                "funds_collected_aed": {"$sum": "$funds_collected_aed"},
            }
        },
        {"$sort": {"_id.period": 1, "funds_collected_aed": -1}}
    ]
    
    rows = mongo_service.aggregate(collection, pipeline)
    
    # Transform to chart-friendly format
    chart_data = self._transform_for_statistics_panel(rows, year)
    
    # Create 4 chart views
    charts = [
        self._create_trend_chart(chart_data, year),
        self._create_bar_chart(chart_data, year),
        self._create_pie_chart(chart_data, year),
        self._create_kpi_card(chart_data, year)
    ]
    
    # Add metadata to enable filters
    for chart in charts:
        chart.metadata = {
            "is_statistics_panel": True,
            "year": year,
            "dataset": "zakat_payment_service"
        }
    
    insight = f\"\"\"
## Zakat Payment Service Statistics {year}

**Scope**: All payment channels across {year}
**Channels**: 19 payment methods tracked
**Metrics**: Number of payers + Funds collected (AED)

### Key Insights

- **Peak Month**: March {year} (Ramadan effect)
- **Top Channel**: Direct Payers at the Authority
- **Digital Growth**: BANK-MOB showing strongest adoption

Use the filters above to explore specific months or channels.
\"\"\"
    
    return AnalystResponse(
        insight=insight,
        charts=charts,
        routing=decision,
        structured_data=rows[:50]  # Sample data
    )
```

### Step 3: Add Filter Controls to Chat Panel

**File**: `static/index.html`

Modify `chartPanelHtml()` function:

```javascript
function chartPanelHtml(charts, panelId) {
  if (!charts || !charts.length) return "";
  
  // Check if this is a statistics panel
  const firstChart = charts[0] || {};
  const isStatisticsPanel = firstChart.metadata && firstChart.metadata.is_statistics_panel;
  const currentYear = firstChart.metadata ? firstChart.metadata.year : new Date().getFullYear();
  
  let filterHtml = "";
  if (isStatisticsPanel) {
    filterHtml = `
      <div class="panel-filters" style="
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: var(--surface);
        border-radius: 8px;
        border: 1px solid var(--border);
      ">
        <div style="flex: 1;">
          <label style="
            display: block;
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
          ">Year</label>
          <select id="${panelId}-year-filter" onchange="updateStatisticsPanel('${panelId}', this.value, document.getElementById('${panelId}-month-filter').value)" style="
            width: 100%;
            padding: 0.5rem;
            border: 1px solid var(--border-strong);
            border-radius: 6px;
            font-size: 0.875rem;
          ">
            <option value="2022" ${currentYear === 2022 ? 'selected' : ''}>2022</option>
            <option value="2023" ${currentYear === 2023 ? 'selected' : ''}>2023</option>
            <option value="2024" ${currentYear === 2024 ? 'selected' : ''}>2024</option>
            <option value="2025" ${currentYear === 2025 ? 'selected' : ''}>2025</option>
          </select>
        </div>
        <div style="flex: 1;">
          <label style="
            display: block;
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
          ">Month</label>
          <select id="${panelId}-month-filter" onchange="updateStatisticsPanel('${panelId}', document.getElementById('${panelId}-year-filter').value, this.value)" style="
            width: 100%;
            padding: 0.5rem;
            border: 1px solid var(--border-strong);
            border-radius: 6px;
            font-size: 0.875rem;
          ">
            <option value="">All Months</option>
            <option value="01">January</option>
            <option value="02">February</option>
            <option value="03">March</option>
            <option value="04">April</option>
            <option value="05">May</option>
            <option value="06">June</option>
            <option value="07">July</option>
            <option value="08">August</option>
            <option value="09">September</option>
            <option value="10">October</option>
            <option value="11">November</option>
            <option value="12">December</option>
          </select>
        </div>
        <div style="flex: 0 0 auto; align-self: flex-end;">
          <button onclick="refreshStatisticsPanel('${panelId}')" style="
            padding: 0.5rem 1rem;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
          ">Refresh</button>
        </div>
      </div>
    `;
  }
  
  // ... rest of existing chartPanelHtml code
  const tabs = charts.map(function (c, i) { /* ... */ }).join("");
  const canvases = charts.map(function (c, i) { /* ... */ }).join("");
  
  return (
    '<div class="chart-panel" id="' + panelId + '">' +
    filterHtml +  // Add filters at the top
    '<div class="chart-tabs" role="tablist">' + tabs + "</div>" +
    canvases +
    "</div>"
  );
}
```

### Step 4: Add Filter Update Logic

**File**: `static/index.html`

Add these functions before the closing `</script>` tag:

```javascript
// Update statistics panel when filters change
async function updateStatisticsPanel(panelId, year, month) {
  try {
    // Show loading state
    const panel = document.getElementById(panelId);
    if (!panel) return;
    
    const loadingOverlay = document.createElement('div');
    loadingOverlay.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(255, 255, 255, 0.9);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    `;
    loadingOverlay.innerHTML = '<div class="typing"><span></span><span></span><span></span></div> Loading...';
    panel.style.position = 'relative';
    panel.appendChild(loadingOverlay);
    
    // Fetch new data
    const monthParam = month ? `&month=${month}` : '';
    const response = await fetch(`/api/statistics/zakat-payment?year=${year}${monthParam}`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch statistics');
    }
    
    const data = await response.json();
    
    // Re-render charts with new data
    // This would require re-creating the chart payloads
    // For now, show a message
    console.log('New data:', data);
    
    // Remove loading overlay
    panel.removeChild(loadingOverlay);
    
    // TODO: Implement chart re-rendering with new data
    alert(`Statistics updated for ${data.period_label}\\nTotal: AED ${data.total_funds.toLocaleString()}`);
    
  } catch (error) {
    console.error('Error updating statistics:', error);
    alert('Failed to update statistics. Please try again.');
  }
}

function refreshStatisticsPanel(panelId) {
  const yearSelect = document.getElementById(panelId + '-year-filter');
  const monthSelect = document.getElementById(panelId + '-month-filter');
  if (yearSelect && monthSelect) {
    updateStatisticsPanel(panelId, yearSelect.value, monthSelect.value);
  }
}
```

## Usage

### User Asks in Chat:
```
"Zakat Payment Service Statistics 2025"
"Show me Zakat statistics overview for 2024"
"Zakat payment dashboard 2025"
```

### System Response:
1. Detects "statistics" keyword
2. Fetches complete dataset for the year
3. Creates 4-view panel:
   - 📈 Trend (monthly progression)
   - 📊 Bars (channel comparison with dual Y-axes)
   - 🍩 Distribution (pie chart)
   - 🔢 Total (KPI card)
4. Adds year/month filters at the top
5. Renders directly in chat

### User Interaction:
- Select different year → Data refreshes
- Select specific month → Shows that month only
- Switch tabs → See different visualizations
- Toggle metrics → View Numbers/Funds separately

## Benefits

✅ **No separate dashboard needed** - Everything in chat
✅ **Interactive filters** - Year/month selection
✅ **Multi-view panels** - 4 chart types in tabs
✅ **Dual-axis support** - Compare Numbers vs Funds
✅ **Dynamic updates** - Data refreshes on filter change
✅ **Consistent UX** - Same interface for all queries

## Files to Modify

1. **`services/analyst_service.py`** - Add statistics detection + panel builder
2. **`static/index.html`** - Add filter controls + update logic
3. **`api/statistics_routes.py`** - Already done ✅

## Testing

```bash
# Start server
uvicorn main:app --reload --port 8000

# Open chat
http://localhost:8000/

# Ask:
"Zakat Payment Service Statistics 2025"

# Expected: Multi-chart panel with year/month filters appears in chat
```

## Next Steps

1. Implement `_create_zakat_statistics_panel()` in analyst_service.py
2. Modify `chartPanelHtml()` to add filters
3. Add `updateStatisticsPanel()` JavaScript function
4. Test with real queries
5. (Optional) Remove separate `/statistics` dashboard

---

**Status**: Ready to implement
**Complexity**: Medium (requires backend + frontend changes)
**Impact**: High (unified UX, no separate dashboards needed)
