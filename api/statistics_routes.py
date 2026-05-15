"""Statistics API endpoints for dynamic data visualization.

Provides dedicated endpoints for statistics dashboards with year/month filtering.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.mongo_service import mongo_service
from utils.logger import logger

router = APIRouter(prefix="/api/statistics", tags=["statistics"])


class ChannelStatistics(BaseModel):
    """Statistics for a single payment channel."""
    channel: str = Field(..., description="Payment channel name")
    number_of_payers: int = Field(0, description="Count of payers")
    funds_collected_aed: float = Field(0.0, description="Funds collected in AED")


class ZakatStatisticsResponse(BaseModel):
    """Response model for Zakat payment statistics."""
    year: int = Field(..., description="Year of data")
    month: str | None = Field(None, description="Month filter (01-12) or None for all")
    channels: list[ChannelStatistics] = Field(..., description="Statistics by channel")
    total_payers: int = Field(..., description="Total number of payers")
    total_funds: float = Field(..., description="Total funds collected in AED")
    period_label: str = Field(..., description="Human-readable period description")


@router.get("/zakat-payment", response_model=ZakatStatisticsResponse)
def get_zakat_payment_statistics(
    year: int = Query(..., description="Year to filter (e.g., 2025)", ge=2020, le=2030),
    month: str | None = Query(None, description="Month to filter (01-12), or omit for all months", regex="^(0[1-9]|1[0-2])$"),
) -> ZakatStatisticsResponse:
    """
    Get Zakat payment service statistics by year and optional month.
    
    Returns aggregated data by payment channel with dual metrics:
    - number_of_payers: Count of individual payers
    - funds_collected_aed: Total funds collected in AED
    
    Examples:
    - /api/statistics/zakat-payment?year=2025 → All months in 2025
    - /api/statistics/zakat-payment?year=2025&month=03 → March 2025 only
    """
    try:
        collection = "awqaf_zakat_payment_service_facts"
        
        # Build MongoDB aggregation pipeline
        match_stage: dict[str, Any] = {"year": year}
        
        if month:
            # Match specific month (e.g., "2025-03")
            period = f"{year}-{month}"
            match_stage["period"] = period
            period_label = f"{_month_name(month)} {year}"
        else:
            # Match all months in the year
            period_label = f"All Months {year}"
        
        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": "$dimension",  # Group by channel
                    "number_of_payers": {"$sum": "$number_of_payers"},
                    "funds_collected_aed": {"$sum": "$funds_collected_aed"},
                }
            },
            {"$sort": {"funds_collected_aed": -1}},  # Sort by funds descending
        ]
        
        logger.info(f"Fetching Zakat statistics: year={year}, month={month}, collection={collection}")
        
        # Execute aggregation (synchronous call)
        results = mongo_service.aggregate(collection, pipeline)
        
        if not results:
            logger.warning(f"No data found for year={year}, month={month}")
            # Return empty response instead of 404 for better UX
            return ZakatStatisticsResponse(
                year=year,
                month=month,
                channels=[],
                total_payers=0,
                total_funds=0.0,
                period_label=period_label,
            )
        
        # Transform results
        channels = [
            ChannelStatistics(
                channel=row["_id"] or "(Unknown)",
                number_of_payers=row.get("number_of_payers", 0),
                funds_collected_aed=row.get("funds_collected_aed", 0.0),
            )
            for row in results
        ]
        
        # Calculate totals
        total_payers = sum(c.number_of_payers for c in channels)
        total_funds = sum(c.funds_collected_aed for c in channels)
        
        logger.info(
            f"Zakat statistics retrieved: {len(channels)} channels, "
            f"{total_payers:,} payers, AED {total_funds:,.2f}"
        )
        
        return ZakatStatisticsResponse(
            year=year,
            month=month,
            channels=channels,
            total_payers=total_payers,
            total_funds=total_funds,
            period_label=period_label,
        )
        
    except Exception as e:
        logger.error(f"Error fetching Zakat statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get("/zakat-payment/trend", response_model=dict[str, Any])
def get_zakat_payment_trend(
    year: int = Query(..., description="Year to analyze", ge=2020, le=2030),
    channel: str | None = Query(None, description="Specific channel to track, or omit for top channels"),
) -> dict[str, Any]:
    """
    Get monthly trend data for Zakat payments across the year.
    
    Returns time-series data suitable for line/trend charts.
    
    Examples:
    - /api/statistics/zakat-payment/trend?year=2025 → Top channels monthly trend
    - /api/statistics/zakat-payment/trend?year=2025&channel=BANK-MOB → BANK-MOB trend
    """
    try:
        collection = "awqaf_zakat_payment_service_facts"
        
        match_stage: dict[str, Any] = {"year": year}
        if channel:
            match_stage["dimension"] = channel
        
        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": {
                        "period": "$period",
                        "channel": "$dimension",
                    },
                    "number_of_payers": {"$sum": "$number_of_payers"},
                    "funds_collected_aed": {"$sum": "$funds_collected_aed"},
                }
            },
            {"$sort": {"_id.period": 1}},
        ]
        
        results = mongo_service.aggregate(collection, pipeline)
        
        # Transform to time-series format
        trend_data: dict[str, Any] = {
            "year": year,
            "channel_filter": channel,
            "series": {},
        }
        
        for row in results:
            channel_name = row["_id"]["channel"] or "(Unknown)"
            period = row["_id"]["period"]
            
            if channel_name not in trend_data["series"]:
                trend_data["series"][channel_name] = []
            
            trend_data["series"][channel_name].append({
                "period": period,
                "number_of_payers": row.get("number_of_payers", 0),
                "funds_collected_aed": row.get("funds_collected_aed", 0.0),
            })
        
        return trend_data
        
    except Exception as e:
        logger.error(f"Error fetching Zakat trend: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve trend data: {str(e)}"
        )


@router.get("/available-years")
def get_available_years() -> dict[str, Any]:
    """
    Get list of years with available Zakat payment data.
    
    Useful for populating year selector dropdowns dynamically.
    """
    try:
        collection = "awqaf_zakat_payment_service_facts"
        
        pipeline = [
            {"$group": {"_id": "$year"}},
            {"$sort": {"_id": -1}},
        ]
        
        results = mongo_service.aggregate(collection, pipeline)
        years = [row["_id"] for row in results if row["_id"]]
        
        return {
            "years": years,
            "latest": years[0] if years else None,
            "count": len(years),
        }
        
    except Exception as e:
        logger.error(f"Error fetching available years: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve available years: {str(e)}"
        )


def _month_name(month_str: str) -> str:
    """Convert month number to name."""
    months = {
        "01": "January", "02": "February", "03": "March",
        "04": "April", "05": "May", "06": "June",
        "07": "July", "08": "August", "09": "September",
        "10": "October", "11": "November", "12": "December",
    }
    return months.get(month_str, month_str)
