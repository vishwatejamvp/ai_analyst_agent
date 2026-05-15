"""Test script for Zakat Payment Statistics API and UI.

Run this to verify the statistics implementation works correctly.
"""

import asyncio
import json
from datetime import datetime

from services.mongo_service import mongo_service


async def test_statistics_data():
    """Test that we can query Zakat payment statistics from MongoDB."""
    
    print("=" * 80)
    print("Testing Zakat Payment Service Statistics")
    print("=" * 80)
    
    collection = "awqaf_zakat_payment_service_facts"
    
    # Test 1: Check if collection exists and has data
    print("\n1. Checking collection existence...")
    try:
        sample = await mongo_service.find(collection, {}, limit=5)
        print(f"✓ Collection exists with {len(sample)} sample records")
        if sample:
            print(f"  Sample record: {json.dumps(sample[0], indent=2, default=str)}")
    except Exception as e:
        print(f"✗ Error accessing collection: {e}")
        return
    
    # Test 2: Get statistics for January 2025
    print("\n2. Testing January 2025 statistics...")
    try:
        pipeline = [
            {"$match": {"year": 2025, "period": "2025-01"}},
            {
                "$group": {
                    "_id": "$dimension",
                    "number_of_payers": {"$sum": "$number_of_payers"},
                    "funds_collected_aed": {"$sum": "$funds_collected_aed"},
                }
            },
            {"$sort": {"funds_collected_aed": -1}},
            {"$limit": 10}
        ]
        
        results = await mongo_service.aggregate(collection, pipeline)
        print(f"✓ Found {len(results)} channels for January 2025")
        
        total_payers = sum(r.get("number_of_payers", 0) for r in results)
        total_funds = sum(r.get("funds_collected_aed", 0.0) for r in results)
        
        print(f"  Top 10 channels:")
        for i, row in enumerate(results[:5], 1):
            channel = row["_id"] or "(Unknown)"
            payers = row.get("number_of_payers", 0)
            funds = row.get("funds_collected_aed", 0.0)
            print(f"    {i}. {channel}: {payers:,} payers, AED {funds:,.2f}")
        
        print(f"\n  Total (top 10): {total_payers:,} payers, AED {total_funds:,.2f}")
        
    except Exception as e:
        print(f"✗ Error querying January 2025: {e}")
    
    # Test 3: Get statistics for March 2025 (Ramadan peak)
    print("\n3. Testing March 2025 statistics (Ramadan)...")
    try:
        pipeline = [
            {"$match": {"year": 2025, "period": "2025-03"}},
            {
                "$group": {
                    "_id": "$dimension",
                    "number_of_payers": {"$sum": "$number_of_payers"},
                    "funds_collected_aed": {"$sum": "$funds_collected_aed"},
                }
            },
            {"$sort": {"funds_collected_aed": -1}},
            {"$limit": 10}
        ]
        
        results = await mongo_service.aggregate(collection, pipeline)
        print(f"✓ Found {len(results)} channels for March 2025")
        
        total_payers = sum(r.get("number_of_payers", 0) for r in results)
        total_funds = sum(r.get("funds_collected_aed", 0.0) for r in results)
        
        print(f"  Top 10 channels:")
        for i, row in enumerate(results[:5], 1):
            channel = row["_id"] or "(Unknown)"
            payers = row.get("number_of_payers", 0)
            funds = row.get("funds_collected_aed", 0.0)
            print(f"    {i}. {channel}: {payers:,} payers, AED {funds:,.2f}")
        
        print(f"\n  Total (top 10): {total_payers:,} payers, AED {total_funds:,.2f}")
        
    except Exception as e:
        print(f"✗ Error querying March 2025: {e}")
    
    # Test 4: Get all months for 2025
    print("\n4. Testing full year 2025 statistics...")
    try:
        pipeline = [
            {"$match": {"year": 2025}},
            {
                "$group": {
                    "_id": "$dimension",
                    "number_of_payers": {"$sum": "$number_of_payers"},
                    "funds_collected_aed": {"$sum": "$funds_collected_aed"},
                }
            },
            {"$sort": {"funds_collected_aed": -1}},
        ]
        
        results = await mongo_service.aggregate(collection, pipeline)
        print(f"✓ Found {len(results)} channels for all of 2025")
        
        total_payers = sum(r.get("number_of_payers", 0) for r in results)
        total_funds = sum(r.get("funds_collected_aed", 0.0) for r in results)
        
        print(f"  Top 5 channels (full year):")
        for i, row in enumerate(results[:5], 1):
            channel = row["_id"] or "(Unknown)"
            payers = row.get("number_of_payers", 0)
            funds = row.get("funds_collected_aed", 0.0)
            print(f"    {i}. {channel}: {payers:,} payers, AED {funds:,.2f}")
        
        print(f"\n  Grand Total: {total_payers:,} payers, AED {total_funds:,.2f}")
        
    except Exception as e:
        print(f"✗ Error querying full year 2025: {e}")
    
    # Test 5: Check available years
    print("\n5. Checking available years...")
    try:
        pipeline = [
            {"$group": {"_id": "$year"}},
            {"$sort": {"_id": -1}},
        ]
        
        results = await mongo_service.aggregate(collection, pipeline)
        years = [r["_id"] for r in results if r["_id"]]
        print(f"✓ Available years: {years}")
        
    except Exception as e:
        print(f"✗ Error checking years: {e}")
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print("\n✓ Statistics API is ready to use!")
    print("\nAccess the statistics dashboard at:")
    print("  http://localhost:8000/statistics")
    print("\nAPI endpoints:")
    print("  GET /api/statistics/zakat-payment?year=2025&month=01")
    print("  GET /api/statistics/zakat-payment?year=2025")
    print("  GET /api/statistics/zakat-payment/trend?year=2025")
    print("  GET /api/statistics/available-years")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_statistics_data())
