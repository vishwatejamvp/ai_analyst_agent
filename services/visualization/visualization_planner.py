from models.enums import ChartType


class VisualizationPlanner:
    """
    Adaptive visualization planner that selects chart types based on analytical shape.
    
    Priority-based decision tree:
    1. Sparse temporal with dimensions → Grouped categorical bars
    2. Multi-metric categorical → Grouped bars
    3. Strong time trend (3+ points) → Line chart
    4. Weak time trend (2 points) → Categorical bars
    5. Single value → KPI card
    6. Categorical comparison → Bars
    7. Fallback → Bars
    """

    @staticmethod
    def choose(
        shape,
        *,
        has_time: bool,
    ) -> tuple[ChartType, str]:
        """
        Choose the most appropriate chart type for the given analytical shape.
        
        Args:
            shape: AnalyticalShape with dimensional analysis
            has_time: Whether the query has temporal structure
            
        Returns:
            (ChartType, reason) tuple explaining the decision
        """

        # ─────────────────────────────────────────────────────────────
        # PRIORITY 1: SPARSE TEMPORAL PIVOT
        #
        # Only 1 time bucket but multiple dimensions exist.
        # User asked for "trend" but data doesn't support it.
        #
        # Solution: Pivot dimensions to x-axis, metrics to series.
        #
        # Example:
        #   Query: "give trend for Occupancy Rates and Revenues in 2026"
        #   Data: 1 time bucket (2026-01), 7 emirates, 2 metrics
        #   Chart: Grouped bars with emirates on x-axis
        # ─────────────────────────────────────────────────────────────
        if shape.sparse_time_series and shape.categorical_comparison:
            if shape.multi_metric_comparison:
                return (
                    ChartType.BAR,
                    (
                        f"Sparse temporal dataset (1 time bucket, "
                        f"{shape.distinct_groups} categories, "
                        f"{shape.distinct_metrics} metrics); "
                        f"rendering grouped categorical comparison."
                    )
                )
            else:
                return (
                    ChartType.BAR,
                    (
                        f"Sparse temporal dataset (1 time bucket, "
                        f"{shape.distinct_groups} categories); "
                        f"rendering categorical comparison."
                    )
                )

        # ─────────────────────────────────────────────────────────────
        # PRIORITY 2: MULTI-METRIC CATEGORICAL COMPARISON
        #
        # No meaningful time axis, but comparing multiple metrics
        # across categories.
        #
        # Example:
        #   Query: "compare revenues and occupancy by emirate"
        #   Data: No time, 7 emirates, 2 metrics
        #   Chart: Grouped bars
        # ─────────────────────────────────────────────────────────────
        if shape.multi_metric_comparison and shape.categorical_comparison:
            return (
                ChartType.BAR,
                (
                    f"Multi-metric categorical comparison "
                    f"({shape.distinct_metrics} metrics across "
                    f"{shape.distinct_groups} categories)."
                )
            )

        # ─────────────────────────────────────────────────────────────
        # PRIORITY 3: STRONG TIME TREND
        #
        # 3+ time points with temporal structure.
        #
        # Example:
        #   Query: "monthly revenues in 2025"
        #   Data: 12 months
        #   Chart: Line trend
        # ─────────────────────────────────────────────────────────────
        if has_time and shape.strong_trend:
            if shape.distinct_series > 1:
                return (
                    ChartType.LINE,
                    (
                        f"Multi-series time trend ({shape.distinct_series} series, "
                        f"{shape.distinct_time_points} time points)."
                    )
                )
            else:
                return (
                    ChartType.LINE,
                    f"Time trend ({shape.distinct_time_points} time points)."
                )

        # ─────────────────────────────────────────────────────────────
        # PRIORITY 4: WEAK TIME TREND → CATEGORICAL
        #
        # 2 time points is ambiguous (could be comparison, not trend).
        # Render as categorical to avoid misleading "trend" appearance.
        #
        # Example:
        #   Query: "revenues in Q1 and Q2"
        #   Data: 2 quarters
        #   Chart: Categorical bars (not line)
        # ─────────────────────────────────────────────────────────────
        if has_time and shape.distinct_time_points == 2:
            return (
                ChartType.BAR,
                (
                    "Insufficient temporal density (2 points); "
                    "rendering as categorical comparison."
                )
            )

        # ─────────────────────────────────────────────────────────────
        # PRIORITY 5: SINGLE VALUE → KPI
        #
        # Only 1 data point total.
        #
        # Example:
        #   Query: "total revenues in 2026"
        #   Data: 1 aggregated value
        #   Chart: KPI card
        # ─────────────────────────────────────────────────────────────
        if shape.row_count == 1:
            return (
                ChartType.KPI,
                "Single data point; rendering as KPI card."
            )

        # ─────────────────────────────────────────────────────────────
        # PRIORITY 6: CATEGORICAL COMPARISON
        #
        # Multiple categories, no time structure.
        #
        # Example:
        #   Query: "revenues by emirate"
        #   Data: 7 emirates
        #   Chart: Categorical bars
        # ─────────────────────────────────────────────────────────────
        if shape.comparison_ready:
            return (
                ChartType.BAR,
                f"Categorical comparison ({shape.distinct_groups} categories)."
            )

        # ─────────────────────────────────────────────────────────────
        # FALLBACK: DEFAULT TO BARS
        #
        # When no other pattern matches, bars are the safest default.
        # ─────────────────────────────────────────────────────────────
        return (
            ChartType.BAR,
            f"Default categorical rendering ({shape.row_count} rows)."
        )