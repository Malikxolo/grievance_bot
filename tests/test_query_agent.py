"""
COMPREHENSIVE MongoDB Query Agent Test - FIXED VERSION
=======================================================

Tests ALL 29 MongoDB MCP tools with proper validation.
This version FIXES:
- Connection verification
- String slicing bug
- Tool selection validation
- False positive detection

⚠️  WARNING: Tests dangerous operations including:
   - delete-many, drop-collection, drop-database
   - atlas operations (may create clusters/users)

Database: comprehensive_test_db
Collections: products, backup_products, users
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from core.mcp.mongodb import MongoDBMCPClient
from core.mcp.query_agent import QueryAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult:
    def __init__(self, name, category, expected_tool=None):
        self.name = name
        self.category = category
        self.expected_tool = expected_tool  # What tool SHOULD be used
        self.actual_tool = None  # What tool WAS used
        self.passed = False
        self.error = None
        self.notes = None
        self.wrong_tool = False  # Flag if wrong tool was selected


def safe_truncate(value: Any, max_length: int = 200) -> str:
    """Safely truncate any value to string"""
    if value is None:
        return "None"
    
    value_str = str(value)
    if len(value_str) <= max_length:
        return value_str
    return value_str[:max_length] + "..."


async def run_test(
    test_name: str,
    instruction: str,
    query_agent: QueryAgent,
    tools_prompt: str,
    mongodb_client: MongoDBMCPClient,
    category: str = "SAFE",
    expected_tool: Optional[str] = None
) -> TestResult:
    """Run a single test with proper validation"""
    
    print(f"\n{'='*70}")
    print(f"🔹 {test_name}")
    print(f"   Category: {category}")
    if expected_tool:
        print(f"   Expected Tool: {expected_tool}")
    print(f"{'='*70}")
    print(f"💬 User: \"{instruction}\"")
    
    test_result = TestResult(test_name, category, expected_tool)
    
    try:
        result = await query_agent.execute(
            tools_prompt=tools_prompt,
            instruction=instruction,
            mcp_client=mongodb_client
        )
        
        # Record actual tool used
        test_result.actual_tool = result.tool_name
        
        # Check if wrong tool was selected
        if expected_tool and result.tool_name and result.tool_name != expected_tool:
            test_result.wrong_tool = True
            print(f"   ⚠️  WRONG TOOL: Expected '{expected_tool}', got '{result.tool_name}'")
        
        if result.needs_clarification:
            print(f"   ❓ Needs clarification: {result.clarification_message}")
            test_result.passed = False
            test_result.error = f"Needs clarification: {result.clarification_message}"
            test_result.notes = "May need more specific parameters"
            
        elif result.success:
            # Check if result actually contains errors (false positive detection)
            result_str = safe_truncate(result.result, 500).lower()
            
            error_patterns = [
                "you need to connect",
                "mcp error",
                "validation error",
                "invalid arguments",
                "not authenticated",
                "ns does not exist"
            ]
            
            has_error = any(pattern in result_str for pattern in error_patterns)
            
            if has_error:
                print(f"   ❌ FALSE SUCCESS - Result contains error:")
                print(f"      {safe_truncate(result.result, 200)}")
                test_result.passed = False
                test_result.error = f"Operation returned error: {safe_truncate(result.result, 100)}"
            elif test_result.wrong_tool:
                print(f"   ⚠️  SUCCESS but used WRONG TOOL")
                print(f"      Result: {safe_truncate(result.result, 200)}")
                test_result.passed = False  # Mark as fail if wrong tool
                test_result.error = f"Wrong tool: expected {expected_tool}, used {result.tool_name}"
            else:
                print(f"   ✅ Success!")
                print(f"      Tool: {result.tool_name}")
                print(f"      Result: {safe_truncate(result.result, 200)}")
                test_result.passed = True
        else:
            print(f"   ❌ Failed: {result.error}")
            test_result.passed = False
            test_result.error = result.error
        
        return test_result
        
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        test_result.passed = False
        test_result.error = str(e)
        return test_result


async def verify_connection(mongodb_client: MongoDBMCPClient) -> bool:
    """Verify MongoDB is actually connected by listing databases"""
    try:
        result = await mongodb_client.execute_tool("list-databases", {})
        
        if hasattr(result, 'result'):
            result_str = str(result.result).lower()
            if "you need to connect" in result_str:
                return False
            return True
        return False
    except Exception as e:
        logger.error(f"Connection verification failed: {e}")
        return False


async def test_all_tools():
    """Comprehensive test of ALL 29 MongoDB MCP tools with proper validation"""
    
    print("\n" + "═"*70)
    print("║                                                                    ║")
    print("║     COMPREHENSIVE MONGODB QUERY AGENT TEST (FIXED)                ║")
    print("║     Testing ALL 29 Tools with Proper Validation                  ║")
    print("║                                                                    ║")
    print("═"*70 + "\n")
    
    load_dotenv()
    
    if not os.getenv("MONGODB_MCP_CONNECTION_STRING"):
        print("❌ MONGODB_MCP_CONNECTION_STRING not set in .env")
        return
    
    # Initialize
    print("📦 Initializing MongoDB MCP Client and Query Agent...\n")
    mongodb_client = MongoDBMCPClient()
    
    try:
        connected = await mongodb_client.connect()
        if not connected:
            print("❌ Connection failed")
            return
        print("✅ MongoDB MCP server started\n")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # CRITICAL: Verify actual database connection
    print("🔍 Verifying database connection...")
    is_connected = await verify_connection(mongodb_client)
    
    if not is_connected:
        print("❌ Database not connected! Trying explicit connect...")
        try:
            connect_result = await mongodb_client.execute_tool("connect", {
                "connectionString": os.getenv("MONGODB_MCP_CONNECTION_STRING")
            })
            print(f"   Connect result: {connect_result}")
            
            # Verify again
            is_connected = await verify_connection(mongodb_client)
            if not is_connected:
                print("❌ Still not connected. Tests may fail.")
            else:
                print("✅ Database connected successfully!")
        except Exception as e:
            print(f"❌ Connect failed: {e}")
    else:
        print("✅ Database connection verified!\n")
    
    query_agent = QueryAgent()
    tools_prompt = mongodb_client.get_tools_prompt()
    print("✅ Query Agent initialized\n")
    
    # Test configuration
    DB_NAME = "comprehensive_test_db"
    COLLECTION_MAIN = "products"
    COLLECTION_BACKUP = "backup_products"
    COLLECTION_USERS = "users"
    
    print("═"*70)
    print(f"📊 Test Configuration:")
    print(f"   Database: {DB_NAME}")
    print(f"   Collections: {COLLECTION_MAIN}, {COLLECTION_BACKUP}, {COLLECTION_USERS}")
    print("═"*70 + "\n")
    
    results = []
    
    # ========================================================================
    # PHASE 1: CONNECTION & SETUP (2 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 1: CONNECTION & SETUP (2 tests)")
    print("█"*70)
    
    test_result = TestResult("Connect to MongoDB", "SAFE", "connect")
    test_result.passed = is_connected
    test_result.notes = "Verified during initialization"
    results.append(test_result)
    print(f"\n{'✅' if is_connected else '❌'} TEST 1: Connect to MongoDB - {'Connected' if is_connected else 'FAILED'}")
    
    results.append(await run_test(
        "List All Databases",
        "Show me all databases in MongoDB",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="list-databases"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 2: BASIC READ OPERATIONS (6 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 2: BASIC READ OPERATIONS (6 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "List Collections in Database",
        f"Show me all collections in {DB_NAME} database",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="list-collections"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "View Collection Schema",
        f"What is the schema of {COLLECTION_MAIN} collection in {DB_NAME}?",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="collection-schema"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "View Collection Indexes",
        f"Show me the indexes on {COLLECTION_MAIN} collection in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="collection-indexes"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Count Documents (Empty)",
        f"How many documents are in {COLLECTION_MAIN} collection in {DB_NAME}?",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="count"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Find Documents (Empty)",
        f"Show me all documents in {COLLECTION_MAIN} collection from {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="find"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Explain Query Plan",
        f"Explain how MongoDB would execute a find query on {COLLECTION_MAIN} in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="explain"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 3: WRITE OPERATIONS - INSERT (3 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 3: WRITE OPERATIONS - INSERT (3 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "Insert One Document",
        f"Add a product named laptop with price 1200 to {COLLECTION_MAIN} collection in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="insert-one"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Insert Many Documents",
        f"Add mouse, keyboard, and monitor to {COLLECTION_MAIN} in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="insert-many"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Insert to Different Collection",
        f"Add user named John with email john@test.com to {COLLECTION_USERS} in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="insert-one"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 4: READ OPERATIONS WITH DATA (3 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 4: READ OPERATIONS WITH DATA (3 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "Find Specific Document",
        f"Find the laptop product in {COLLECTION_MAIN} collection in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="find"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Count Documents (With Data)",
        f"How many products are in {COLLECTION_MAIN} in {DB_NAME}?",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="count"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Aggregation Pipeline",
        f"In {DB_NAME}, use aggregation to count how many products have a price field in {COLLECTION_MAIN}",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="aggregate"  # THIS should use aggregate, not count!
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 5: UPDATE OPERATIONS (2 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 5: UPDATE OPERATIONS (2 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "Update One Document",
        f"Update the laptop in {COLLECTION_MAIN} in {DB_NAME} to set brand as Dell",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="update-one"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Update Many Documents",
        f"Update all products in {COLLECTION_MAIN} in {DB_NAME} to add a category field with value electronics",
        query_agent, tools_prompt, mongodb_client,
        category="MODERATE_RISK",
        expected_tool="update-many"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 6: INDEX OPERATIONS (1 test - DANGEROUS)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 6: INDEX OPERATIONS (1 test - DANGEROUS)")
    print("█"*70)
    
    results.append(await run_test(
        "Create Index",
        f"Create an index on the name field of {COLLECTION_MAIN} collection in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="DANGEROUS",
        expected_tool="create-index"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 7: ADMIN OPERATIONS (2 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 7: ADMIN OPERATIONS (2 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "Database Statistics",
        f"Show me statistics for {DB_NAME} database",
        query_agent, tools_prompt, mongodb_client,
        category="ADMIN",
        expected_tool="db-stats"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Collection Storage Size",
        f"What is the storage size of {COLLECTION_MAIN} in {DB_NAME}?",
        query_agent, tools_prompt, mongodb_client,
        category="ADMIN",
        expected_tool="collection-storage-size"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 8: DELETE OPERATIONS (2 tests - DANGEROUS)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 8: DELETE OPERATIONS (2 tests - DANGEROUS)")
    print("█"*70)
    
    results.append(await run_test(
        "Delete One Document",
        f"Delete the mouse product from {COLLECTION_MAIN} in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="SAFE",
        expected_tool="delete-one"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Delete Many Documents (DANGEROUS)",
        f"Delete all products with category electronics from {COLLECTION_MAIN} in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="DANGEROUS",
        expected_tool="delete-many"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 9: COLLECTION MANAGEMENT (2 tests - DANGEROUS)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 9: COLLECTION MANAGEMENT (2 tests - DANGEROUS)")
    print("█"*70)
    
    results.append(await run_test(
        "Rename Collection (DANGEROUS)",
        f"Rename {COLLECTION_MAIN} to {COLLECTION_BACKUP} in {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="DANGEROUS",
        expected_tool="rename-collection"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Drop Collection (DANGEROUS)",
        f"Delete the entire {COLLECTION_USERS} collection from {DB_NAME}",
        query_agent, tools_prompt, mongodb_client,
        category="DANGEROUS",
        expected_tool="drop-collection"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 10: ATLAS MANAGEMENT (6 tests - ADMIN/DANGEROUS)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 10: ATLAS MANAGEMENT (6 tests - ADMIN/DANGEROUS)")
    print("█"*70)
    
    results.append(await run_test(
        "List Atlas Clusters",
        "Show me all MongoDB Atlas clusters",
        query_agent, tools_prompt, mongodb_client,
        category="ADMIN",
        expected_tool="atlas-list-clusters"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "List Atlas Projects",
        "Show me all MongoDB Atlas projects",
        query_agent, tools_prompt, mongodb_client,
        category="ADMIN",
        expected_tool="atlas-list-projects"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Inspect Atlas Cluster",
        "Show me details of my current MongoDB Atlas cluster",
        query_agent, tools_prompt, mongodb_client,
        category="ADMIN",
        expected_tool="atlas-inspect-cluster"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Inspect Access List",
        "Show me the IP access list for my Atlas project",
        query_agent, tools_prompt, mongodb_client,
        category="ADMIN",
        expected_tool="atlas-inspect-access-list"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "List Database Users",
        "Show me all database users in my Atlas project",
        query_agent, tools_prompt, mongodb_client,
        category="ADMIN",
        expected_tool="atlas-list-db-users"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Create Database User (DANGEROUS)",
        "Create a database user named testuser with password test123 with read role in my Atlas project",
        query_agent, tools_prompt, mongodb_client,
        category="DANGEROUS",
        expected_tool="atlas-create-db-user"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 11: MOST DANGEROUS OPERATIONS (3 tests - VERY DANGEROUS)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 11: MOST DANGEROUS OPERATIONS (3 tests - VERY DANGEROUS)")
    print("█  ⚠️  WARNING: These operations can cause data loss or cost money!")
    print("█"*70)
    
    results.append(await run_test(
        "Create Access List Entry (DANGEROUS)",
        "Add IP address 203.0.113.0/24 to the Atlas access list for my project",
        query_agent, tools_prompt, mongodb_client,
        category="VERY_DANGEROUS",
        expected_tool="atlas-create-access-list"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Drop Database (VERY DANGEROUS)",
        f"Delete the entire {DB_NAME} database and all its data",
        query_agent, tools_prompt, mongodb_client,
        category="VERY_DANGEROUS",
        expected_tool="drop-database"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "Create Free Cluster (VERY DANGEROUS)",
        "Create a new free MongoDB Atlas cluster named test-cluster in AWS us-east-1",
        query_agent, tools_prompt, mongodb_client,
        category="VERY_DANGEROUS",
        expected_tool="atlas-create-free-cluster"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "═"*70)
    print("█                                                                    █")
    print("█                        TEST SUMMARY                               █")
    print("█                                                                    █")
    print("═"*70 + "\n")
    
    # Categorize results
    by_category = {}
    wrong_tool_tests = []
    
    for result in results:
        if result.category not in by_category:
            by_category[result.category] = []
        by_category[result.category].append(result)
        
        if result.wrong_tool:
            wrong_tool_tests.append(result)
    
    # Print by category
    category_order = ["SAFE", "MODERATE_RISK", "ADMIN", "DANGEROUS", "VERY_DANGEROUS"]
    
    for category in category_order:
        if category not in by_category:
            continue
        
        tests = by_category[category]
        passed = sum(1 for t in tests if t.passed)
        total = len(tests)
        
        print(f"\n{'='*70}")
        print(f"📊 {category} TOOLS: {passed}/{total} passed")
        print(f"{'='*70}")
        
        for test in tests:
            status = "✅ PASS" if test.passed else "❌ FAIL"
            tool_info = f" [Expected: {test.expected_tool}, Used: {test.actual_tool}]" if test.expected_tool else ""
            print(f"{status} - {test.name}{tool_info}")
            
            if test.wrong_tool:
                print(f"        ⚠️  WRONG TOOL SELECTED!")
            if not test.passed and test.error:
                print(f"        Error: {safe_truncate(test.error, 100)}")
            if test.notes:
                print(f"        Note: {test.notes}")
    
    # Wrong tool summary
    if wrong_tool_tests:
        print(f"\n{'='*70}")
        print(f"⚠️  WRONG TOOL SELECTIONS: {len(wrong_tool_tests)} tests")
        print(f"{'='*70}")
        for test in wrong_tool_tests:
            print(f"❌ {test.name}")
            print(f"   Expected: {test.expected_tool}")
            print(f"   Actually used: {test.actual_tool}")
    
    # Overall stats
    total_tests = len(results)
    total_passed = sum(1 for r in results if r.passed)
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{'═'*70}")
    print(f"OVERALL RESULTS")
    print(f"{'═'*70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_tests - total_passed} ❌")
    print(f"Wrong Tool: {len(wrong_tool_tests)} ⚠️")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"{'═'*70}\n")
    
    # MongoDB stats
    stats = mongodb_client.get_stats()
    print(f"📈 MongoDB Client Stats:")
    print(f"   Total MCP calls: {stats['calls']}")
    print(f"   Successes: {stats['successes']}")
    print(f"   Errors: {stats['errors']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%\n")
    
    # Recommendations
    print(f"{'═'*70}")
    print("💡 RECOMMENDATIONS FOR PRODUCTION")
    print(f"{'═'*70}\n")
    
    print("✅ ALLOW (Safe and working correctly):")
    safe_tests = [t for t in by_category.get("SAFE", []) if t.passed and not t.wrong_tool]
    for test in safe_tests:
        print(f"   ✓ {test.expected_tool or test.name}")
    
    print("\n⚠️  NEEDS IMPROVEMENT (Wrong tool selection):")
    for test in wrong_tool_tests:
        print(f"   ! {test.name} - Query Agent needs better training")
    
    print("\n🚫 BLOCK COMPLETELY:")
    dangerous_tests = by_category.get("DANGEROUS", []) + by_category.get("VERY_DANGEROUS", [])
    for test in dangerous_tests:
        if test.passed:
            print(f"   ❌ {test.expected_tool or test.name}")
    
    print(f"\n{'═'*70}")
    print("🔍 NEXT STEPS:")
    print(f"{'═'*70}")
    print("1. Fix Query Agent prompt to select correct tools (especially aggregate)")
    print("2. Add tool validation before execution")
    print("3. Implement tool filtering in mongodb.py (block dangerous tools)")
    print("4. Add connection verification in all MCP clients")
    print(f"{'═'*70}\n")
    
    # Properly cleanup resources
    await query_agent.close()
    await mongodb_client.disconnect()
    print("[OK] Test completed and disconnected\n")


if __name__ == "__main__":
    print("""
================================================================
       COMPREHENSIVE MONGODB QUERY AGENT TEST (FIXED)             
       Tests ALL 29 Tools with Proper Validation                 
                                                                  
  FIXES:                                                          
  [OK] Connection verification                                      
  [OK] String slicing bug                                           
  [OK] Tool selection validation                                    
  [OK] False positive detection                                     
                                                                  
  WARNING: Tests DANGEROUS operations!                        
================================================================
    """)
    
    asyncio.run(test_all_tools())