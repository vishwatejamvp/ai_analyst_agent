# Statistics in Chat - Complete Solution

## Problem Summary

**Current State**: When users ask "Zakat Payment Service Statistics 2025" in chat, they get:
- ✅ Multi-chart panel with tabs (Trend, Bars, Distribution, Total)
- ✅ Metric selector dropdown
- ❌ NO year/month filters
- ❌ Metric selector shows channel names instead of "All metrics / Numbers / Funds"

**Desired State**: Same query should show:
- ✅ Year/Month filter dropdowns above the chart
- ✅ Proper metric toggle (All metrics / Numbers / Funds)
- ✅ Dynamic data refresh when filters change
- ✅ All within the chat interface (no separate dashboard)

## Root Causes

### 1. Missing Filter UI in Chat Panel
**File**: [`static/index.html`](static/index.html:1081) - `chartPanelHtml()` function

The function creates tabs and metric selectors but doesn't include year/month filters. The separate dashboard has these filters hardcoded in its HTML.

### 2. Metric Selector Shows Channels, Not Metrics
**Issue**: The dropdown shows "Funds Collected Aed" repeated 19 times (one per channel).

**Why**: The chart is structured with 19 series (one per channel), so the metric selector treats each channel as a "metric". 

**What's Needed**: The chart should have 2 series (Numbers + Funds), not 19 series (channels).

### 3. No Backend Flag for Statistics Panels
**Issue**: The frontend doesn't know when to show filters.

**Why**: The backend doesn't mark statistics responses differently from regular analytical responses.

**What's Needed**: Add metadata to chart payloads indicating this is a statistics panel.

## Solution Architecture

### Backend Changes

#### 1. Detect Statistics Queries
**File**: `services/analyst_service.py`

```python
def _handle_analytical(self, question: str, decision: RoutingDecision, ...) -> AnalystResponse:
    # Detect statistics keywords
    is_statistics = any(kw in question.lower() for kw in [
        'statistics', 'overview', 'dashboard', 'summary report'
    ])
    
    if is_statistics and 'zakat' in question.lower():
        return self._create_statistics_panel(question, decision)
    
    # ... existing logic
```

#### 2. Create Statistics Panel with Metadata
**File**: `services/analyst_service.py`

```python
def _create_statistics_panel(self, question: str, decision: RoutingDecision) -> AnalystResponse:
    # Extract year from question
    import re
    year_match = re.search(r'20\d{2}', question)
    year = int(year_match.group()) if year_match else 2025
    
    # Fetch data for all channels and months
    collection = "awqaf_zakat_payment_service_facts"
    pipeline = [
        {"$match": {"year": year}},
        {
            "$group": {
                "_id": {"channel": "$dimension", "period": "$period"},
                "number_of_payers": {"$sum": "$number_of_payers"},
                "funds_collected_aed": {"$sum": "$funds_collected_aed"},
            }
        },
        {"$sort": {"_id.period": 1}}
    ]
    
    rows = mongo_service.aggregate(collection, pipeline)
    
    # Create charts using existing chart_service
    # The key is to pass metadata that frontend can detect
    charts = chart_service.render_multi_view_panel(
        rows=rows,
        title=f"Zakat Payment Service Statistics {year}",
        metadata={
            "panel_type": "statistics",
            "year": year,
            "dataset": "zakat_payment_service",
            "supports_filters": True
        }
    )
    
    return AnalystResponse(
        insight=f"## Zakat Payment Service Statistics {year}\\n\\n...",
        charts=charts,
        routing=decision
    )
```

### Frontend Changes

#### 1. Enhance chartPanelHtml() to Add Filters
**File**: `static/index.html` - Line ~1081

```javascript
function chartPanelHtml(charts, panelId) {
  if (!charts || !charts.length) return "";
  
  // Check if this is a statistics panel
  const firstChart = charts[0] || {};
  const metadata = firstChart.metadata || {};
  const isStatisticsPanel = metadata.panel_type === 'statistics';
  const currentYear = metadata.year || new Date().getFullYear();
  
  let filterHtml = "";
  if (isStatisticsPanel) {
    filterHtml = `
      <div class="panel-filters" style="
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: var(--surface);
        border-radius: var(--radius-sm);
        border: 1px solid var(--border);
      ">
        <div style="flex: 1;">
          <label style="
            display: block;
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text);
            margin-bottom: 0.375rem;
          ">Year</label>
          <select 
            id="${panelId}-year-filter" 
            onchange="updateStatisticsPanel('${panelId}')"
            style="
              width: 100%;
              padding: 0.5rem 0.75rem;
              border: 1px solid var(--border-strong);
              border-radius: 6px;
              font-size: 0.875rem;
              background: white;
              cursor: pointer;
            "
          >
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
            color: var(--text);
            margin-bottom: 0.375rem;
          ">Month</label>
          <select 
            id="${panelId}-month-filter" 
            onchange="updateStatisticsPanel('${panelId}')"
            style="
              width: 100%;
              padding: 0.5rem 0.75rem;
              border: 1px solid var(--border-strong);
              border-radius: 6px;
              font-size: 0.875rem;
              background: white;
              cursor: pointer;
            "
          >
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
          <button 
            onclick="updateStatisticsPanel('${panelId}')" 
            style="
              padding: 0.5rem 1.25rem;
              background: var(--accent);
              color: white;
              border: none;
              border-radius: 6px;
              font-size: 0.875rem;
              font-weight: 500;
              cursor: pointer;
              transition: background 0.2s;
            "
            onmouseover="this.style.background='var(--accent-hover)'"
            onmouseout="this.style.background='var(--accent)'"
          >
            Update
          </button>
        </div>
      </div>
    `;
  }
  
  // ... rest of existing code for tabs and canvases
  const tabs = charts.map(function (c, i) {
    // ... existing tab code
  }).join("");
  
  const canvases = charts.map(function (c, i) {
    // ... existing canvas code
  }).join("");
  
  return (
    '<div class="chart-panel" id="' + panelId + '" data-panel-type="' + 
    (isStatisticsPanel ? 'statistics' : 'standard') + '">' +
    filterHtml +  // Filters appear ABOVE tabs
    '<div class="chart-tabs" role="tablist">' + tabs + "</div>" +
    canvases +
    "</div>"
  );
}
```

#### 2. Add Filter Update Function
**File**: `static/index.html` - Add before closing `</script>` tag

```javascript
// Update statistics panel when filters change
async function updateStatisticsPanel(panelId) {
  const panel = document.getElementById(panelId);
  if (!panel || panel.getAttribute('data-panel-type') !== 'statistics') return;
  
  const yearSelect = document.getElementById(panelId + '-year-filter');
  const monthSelect = document.getElementById(panelId + '-month-filter');
  
  if (!yearSelect || !monthSelect) return;
  
  const year = yearSelect.value;
  const month = monthSelect.value;
  
  try {
    // Show loading overlay
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'panel-loading-overlay';
    loadingOverlay.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(255, 255, 255, 0.95);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      border-radius: var(--radius);
    `;
    loadingOverlay.innerHTML = `
      <div class="typing" style="margin-bottom: 0.5rem;">
        <span></span><span></span><span></span>
      </div>
      <div style="font-size: 0.875rem; color: var(--text-muted);">
        Loading ${month ? getMonthName(month) : 'all months'} ${year}...
      </div>
    `;
    panel.style.position = 'relative';
    panel.appendChild(loadingOverlay);
    
    // Fetch new data from API
    const monthParam = month ? `&month=${month}` : '';
    const response = await fetch(`/api/statistics/zakat-payment?year=${year}${monthParam}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    // For now, show summary (full re-rendering requires backend changes)
    const summary = `
      <div style="padding: 1.5rem; text-align: center;">
        <div style="font-size: 2rem; font-weight: 600; color: var(--accent); margin-bottom: 0.5rem;">
          AED ${data.total_funds.toLocaleString()}
        </div>
        <div style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.25rem;">
          ${data.period_label}
        </div>
        <div style="font-size: 0.875rem; color: var(--text-muted);">
          ${data.total_payers.toLocaleString()} payers across ${data.channels.length} channels
        </div>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);">
          <strong>Top 3 Channels:</strong><br/>
          ${data.channels.slice(0, 3).map((ch, i) => 
            `${i + 1}. ${ch.channel}: AED ${ch.funds_collected_aed.toLocaleString()}`
          ).join('<br/>')}
        </div>
      </div>
    `;
    
    // Replace loading with summary
    loadingOverlay.innerHTML = summary;
    loadingOverlay.style.background = 'var(--surface)';
    loadingOverlay.style.border = '1px solid var(--border)';
    loadingOverlay.style.borderRadius = 'var(--radius)';
    loadingOverlay.style.marginTop = '1rem';
    
    // Note: Full chart re-rendering would require re-querying the backend
    // and rebuilding the entire panel. For now, this shows the updated data.
    
  } catch (error) {
    console.error('Error updating statistics:', error);
    const errorMsg = document.createElement('div');
    errorMsg.style.cssText = `
      padding: 1rem;
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #991b1b;
      border-radius: var(--radius-sm);
      margin-top: 1rem;
    `;
    errorMsg.textContent = `Failed to update statistics: ${error.message}`;
    panel.querySelector('.panel-loading-overlay')?.remove();
    panel.appendChild(errorMsg);
    setTimeout(() => errorMsg.remove(), 5000);
  }
}

function getMonthName(monthNum) {
  const months = {
    '01': 'January', '02': 'February', '03': 'March', '04': 'April',
    '05': 'May', '06': 'June', '07': 'July', '08': 'August',
    '09': 'September', '10': 'October', '11': 'November', '12': 'December'
  };
  return months[monthNum] || monthNum;
}
```

## Simplified Implementation (Quick Win)

Since full implementation requires significant backend changes, here's a **quick win** approach:

### Step 1: Add Filters to Existing Chat Panel
Modify `chartPanelHtml()` to always show year/month filters when the chart title contains "Zakat" or "Statistics":

```javascript
// In chartPanelHtml(), after line 1082:
const firstChart = charts[0] || {};
const title = firstChart.view_label || firstChart.chart_type || "";
const isZakatStats = title.toLowerCase().includes('zakat') || 
                     title.toLowerCase().includes('statistics');

if (isZakatStats) {
  // Add filter HTML here
}
```

### Step 2: Make Filters Functional
The `updateStatisticsPanel()` function fetches new data and displays a summary. For full chart re-rendering, users would need to ask a new question like "Show Zakat statistics for March 2024".

## Testing

1. **Start server**: `uvicorn main:app --reload --port 8000`
2. **Open chat**: `http://localhost:8000/`
3. **Ask**: "Zakat Payment Service Statistics 2025"
4. **Expected**: Multi-chart panel with year/month filters above tabs
5. **Test filters**: Change year/month and click "Update"
6. **Expected**: Summary of new data appears

## Current Status

✅ **Completed**:
- Statistics API endpoints working
- Async/await bug fixed
- Separate dashboard removed from header

⏳ **In Progress**:
- Adding filters to chat panel
- Fixing metric selector

❌ **Not Yet Done**:
- Backend detection of statistics queries
- Full chart re-rendering on filter change
- Proper dual-metric structure (Numbers + Funds as 2 series)

## Recommendation

For the **fastest solution**, I recommend:

1. ✅ Add filter UI to chat panel (frontend only)
2. ✅ Show data summary when filters change
3. ⏳ For full chart refresh, guide users to ask new questions

This gives you 80% of the functionality with 20% of the effort. Full dynamic chart re-rendering requires restructuring how charts are created in the backend.

---

**Next Steps**: Implement the filter UI changes in `static/index.html` as shown above.
