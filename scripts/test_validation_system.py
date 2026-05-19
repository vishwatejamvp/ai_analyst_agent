"""Quick validation system test script.

Run this to verify the validation system is working correctly.
"""

from models.schemas import QueryRequest
from services.analyst_service import AnalystOrchestrator

def test_validation():
    """Test validation system with various queries"""
    
    orchestrator = AnalystOrchestrator()
    
    test_cases = [
        {
            "name": "Valid query",
            "question": "total transactions for hajj-permit-service in 2024",
            "expect": "success"
        },
        {
            "name": "Wrong metric",
            "question": "total nonexistent_field for hajj-permit-service in 2024",
            "expect": "warning"
        },
        {
            "name": "Future year",
            "question": "total transactions for hajj-permit-service in 2030",
            "expect": "warning"
        },
        {
            "name": "Semantic matching",
            "question": "how many pilgrimage permits in 2024",
            "expect": "success"
        }
    ]
    
    print("=" * 60)
    print("VALIDATION SYSTEM TEST")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Question: {test['question']}")
        
        try:
            request = QueryRequest(question=test['question'])
            response = orchestrator.answer(request)
            
            print(f"   Target: {response.routing.target}")
            print(f"   Warnings: {len(response.warnings)}")
            
            if response.warnings:
                for w in response.warnings:
                    print(f"     - {w.message[:80]}...")
            
            print(f"   Data rows: {len(response.structured_data)}")
            
            if test['expect'] == "warning" and len(response.warnings) > 0:
                print(f"   ✅ PASS - Warning detected as expected")
            elif test['expect'] == "success" and response.routing.target:
                print(f"   ✅ PASS - Query executed successfully")
            else:
                print(f"   ⚠️  UNEXPECTED RESULT")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_validation()
