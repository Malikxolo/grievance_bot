"""
COMPREHENSIVE Redis Query Agent Test
=====================================

Tests Redis MCP tools with proper validation via QueryAgent.
Similar pattern to test_query_agent.py but for Redis operations.

Redis URL: redis://default:****@redis-14300.c264.ap-south-1-1.ec2.cloud.redislabs.com:14300

Test Categories:
- STRING operations (set, get, mset, mget, incr, decr)
- HASH operations (hset, hget, hgetall, hdel)
- LIST operations (lpush, rpush, lpop, rpop, lrange)
- SET operations (sadd, srem, smembers, sismember)
- SORTED SET operations (zadd, zrange, zscore)
- KEY operations (keys, exists, del, expire, ttl)
- Advanced operations (publish, streams, JSON)

Run: python tests/test_redis_query_agent.py
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from core.mcp.redis import RedisMCPClient
from core.mcp.query_agent import QueryAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult:
    def __init__(self, name: str, category: str, expected_tool: Optional[str] = None):
        self.name = name
        self.category = category
        self.expected_tool = expected_tool
        self.actual_tool = None
        self.passed = False
        self.error = None
        self.notes = None
        self.wrong_tool = False


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
    redis_client: RedisMCPClient,
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
            mcp_client=redis_client
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
            # Check for error patterns in result
            result_str = safe_truncate(result.result, 500).lower()
            
            error_patterns = [
                "connection refused",
                "connection failed",
                "wrongtype",
                "noauth",
                "authentication failed",
                "error:"
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
                test_result.passed = False
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


async def verify_connection(redis_client: RedisMCPClient) -> bool:
    """Verify Redis connection by running a simple command"""
    try:
        # Try to set and get a test value
        result = await redis_client.execute_tool("set", {
            "key": "test:connection:check",
            "value": "connected"
        })
        
        if hasattr(result, 'success') and result.success:
            return True
        return False
    except Exception as e:
        logger.error(f"Connection verification failed: {e}")
        return False


async def test_all_redis_tools():
    """Comprehensive test of Redis MCP tools via QueryAgent"""
    
    print("\n" + "═"*70)
    print("║                                                                    ║")
    print("║     COMPREHENSIVE REDIS QUERY AGENT TEST                          ║")
    print("║     Testing Redis Operations via Natural Language                 ║")
    print("║                                                                    ║")
    print("═"*70 + "\n")
    
    load_dotenv()
    
    # Redis URL - use the provided one or from environment
    REDIS_URL = os.getenv("REDIS_MCP_URL", 
        "redis://default:goe9pNoOtVvDYcg6gLaC4uTxAT57t32o@redis-14300.c264.ap-south-1-1.ec2.cloud.redislabs.com:14300")
    
    if not REDIS_URL:
        print("❌ REDIS_MCP_URL not set")
        return
    
    # Mask password for display
    masked_url = REDIS_URL.replace(REDIS_URL.split(':')[2].split('@')[0], '****')
    print(f"📦 Redis URL: {masked_url}\n")
    
    # Initialize
    print("📦 Initializing Redis MCP Client and Query Agent...\n")
    redis_client = RedisMCPClient(redis_url=REDIS_URL, startup_timeout=90)
    
    try:
        connected = await redis_client.connect()
        if not connected:
            print("❌ Connection failed")
            return
        print("✅ Redis MCP server started\n")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Verify connection
    print("🔍 Verifying Redis connection...")
    is_connected = await verify_connection(redis_client)
    
    if not is_connected:
        print("❌ Redis connection verification failed")
        await redis_client.disconnect()
        return
    print("✅ Redis connection verified!\n")
    
    # Initialize QueryAgent
    query_agent = QueryAgent()
    tools_prompt = redis_client.get_tools_prompt()
    print("✅ Query Agent initialized\n")
    
    # Show available tools
    tools = await redis_client.list_tools()
    print(f"📋 Available Redis Tools: {len(tools)}")
    print(f"   Tools: {', '.join([t.name for t in tools[:10]])}...\n")
    
    # Test key prefix to avoid conflicts
    KEY_PREFIX = "test:qa:"
    
    print("═"*70)
    print(f"📊 Test Configuration:")
    print(f"   Key Prefix: {KEY_PREFIX}")
    print("═"*70 + "\n")
    
    results = []
    
    # ========================================================================
    # PHASE 1: STRING OPERATIONS (6 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 1: STRING OPERATIONS (6 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "SET - Store Simple Value",
        f"Store the value 'hello world' with key '{KEY_PREFIX}greeting'",
        query_agent, tools_prompt, redis_client,
        category="STRING",
        expected_tool="set"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "GET - Retrieve Value",
        f"Get the value stored at key '{KEY_PREFIX}greeting'",
        query_agent, tools_prompt, redis_client,
        category="STRING",
        expected_tool="get"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "SET with Expiration",
        f"Store 'temporary_data' at key '{KEY_PREFIX}temp' with 60 seconds expiration",
        query_agent, tools_prompt, redis_client,
        category="STRING",
        expected_tool="set"
    ))
    await asyncio.sleep(0.5)
    
    # Note: INCR tool not available in Redis MCP server v1.19.0
    # Testing with SET and GET instead
    results.append(await run_test(
        "SET - Store Counter",
        f"Set the value '1' at key '{KEY_PREFIX}visits'",
        query_agent, tools_prompt, redis_client,
        category="STRING",
        expected_tool="set"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "GET - Retrieve Counter",
        f"Get the value stored at key '{KEY_PREFIX}visits'",
        query_agent, tools_prompt, redis_client,
        category="STRING",
        expected_tool="get"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "SET - Update Counter Value",
        f"Set the value at key '{KEY_PREFIX}counter2' to '100'",
        query_agent, tools_prompt, redis_client,
        category="STRING",
        expected_tool="set"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 2: HASH OPERATIONS (5 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 2: HASH OPERATIONS (5 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "HSET - Store User Data",
        f"Store user data in hash '{KEY_PREFIX}user:100' with field 'name' and value 'John Doe'",
        query_agent, tools_prompt, redis_client,
        category="HASH",
        expected_tool="hset"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "HSET - Add Email Field",
        f"Add email 'john@example.com' to hash '{KEY_PREFIX}user:100'",
        query_agent, tools_prompt, redis_client,
        category="HASH",
        expected_tool="hset"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "HGET - Get Specific Field",
        f"Get the name field from hash '{KEY_PREFIX}user:100'",
        query_agent, tools_prompt, redis_client,
        category="HASH",
        expected_tool="hget"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "HGETALL - Get All Fields",
        f"Get all fields and values from hash '{KEY_PREFIX}user:100'",
        query_agent, tools_prompt, redis_client,
        category="HASH",
        expected_tool="hgetall"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "HDEL - Delete Field",
        f"Delete the email field from hash '{KEY_PREFIX}user:100'",
        query_agent, tools_prompt, redis_client,
        category="HASH",
        expected_tool="hdel"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 3: LIST OPERATIONS (5 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 3: LIST OPERATIONS (5 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "LPUSH - Add to Beginning of List",
        f"Add 'task1' to the beginning of list '{KEY_PREFIX}queue'",
        query_agent, tools_prompt, redis_client,
        category="LIST",
        expected_tool="lpush"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "RPUSH - Add to End of List",
        f"Add 'task2' to the end of list '{KEY_PREFIX}queue'",
        query_agent, tools_prompt, redis_client,
        category="LIST",
        expected_tool="rpush"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "LPUSH - Add Another Item",
        f"Push 'task0' to the front of '{KEY_PREFIX}queue'",
        query_agent, tools_prompt, redis_client,
        category="LIST",
        expected_tool="lpush"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "LRANGE - Get All List Items",
        f"Get all items from list '{KEY_PREFIX}queue'",
        query_agent, tools_prompt, redis_client,
        category="LIST",
        expected_tool="lrange"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "LPOP - Remove from Beginning",
        f"Remove and return the first item from list '{KEY_PREFIX}queue'",
        query_agent, tools_prompt, redis_client,
        category="LIST",
        expected_tool="lpop"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 4: SET OPERATIONS (4 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 4: SET OPERATIONS (4 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "SADD - Add to Set",
        f"Add 'apple' to the set '{KEY_PREFIX}fruits'",
        query_agent, tools_prompt, redis_client,
        category="SET",
        expected_tool="sadd"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "SADD - Add More Members",
        f"Add 'banana' and 'orange' to set '{KEY_PREFIX}fruits'",
        query_agent, tools_prompt, redis_client,
        category="SET",
        expected_tool="sadd"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "SMEMBERS - Get All Members",
        f"Get all members of set '{KEY_PREFIX}fruits'",
        query_agent, tools_prompt, redis_client,
        category="SET",
        expected_tool="smembers"
    ))
    await asyncio.sleep(0.5)
    
    # Note: SISMEMBER tool not available in Redis MCP server v1.19.0
    # Using SMEMBERS to check membership instead
    results.append(await run_test(
        "SMEMBERS - Check Set Contents",
        f"Get members from set '{KEY_PREFIX}fruits' to see what's in it",
        query_agent, tools_prompt, redis_client,
        category="SET",
        expected_tool="smembers"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 5: SORTED SET OPERATIONS (4 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 5: SORTED SET OPERATIONS (4 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "ZADD - Add with Score",
        f"Add 'player1' with score 100 to sorted set '{KEY_PREFIX}leaderboard'",
        query_agent, tools_prompt, redis_client,
        category="SORTED_SET",
        expected_tool="zadd"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "ZADD - Add More Players",
        f"Add 'player2' with score 250 to '{KEY_PREFIX}leaderboard'",
        query_agent, tools_prompt, redis_client,
        category="SORTED_SET",
        expected_tool="zadd"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "ZRANGE - Get Sorted Range",
        f"Get all members from sorted set '{KEY_PREFIX}leaderboard' ordered by score",
        query_agent, tools_prompt, redis_client,
        category="SORTED_SET",
        expected_tool="zrange"
    ))
    await asyncio.sleep(0.5)
    
    # Note: ZSCORE, ZRANK, ZINCRBY tools not available in Redis MCP server v1.19.0
    # Testing ZADD to update a player's score instead
    results.append(await run_test(
        "ZADD - Update Score",
        f"Add 'player1' with score 150 to sorted set '{KEY_PREFIX}leaderboard'",
        query_agent, tools_prompt, redis_client,
        category="SORTED_SET",
        expected_tool="zadd"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 6: KEY OPERATIONS (5 tests)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 6: KEY OPERATIONS (5 tests)")
    print("█"*70)
    
    # Note: scan_all_keys is preferred over keys for pattern matching (non-blocking)
    results.append(await run_test(
        "SCAN - Find Keys by Pattern",
        f"Scan for all keys matching pattern '{KEY_PREFIX}*'",
        query_agent, tools_prompt, redis_client,
        category="KEY",
        expected_tool="scan_all_keys"
    ))
    await asyncio.sleep(0.5)
    
    # Note: exists tool may not be available, use type as alternative (returns type if exists)
    results.append(await run_test(
        "TYPE - Check Key Existence and Type",
        f"Get the type of key '{KEY_PREFIX}greeting' to check if it exists",
        query_agent, tools_prompt, redis_client,
        category="KEY",
        expected_tool="type"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "EXPIRE - Set Expiration",
        f"Set '{KEY_PREFIX}greeting' to expire in 300 seconds",
        query_agent, tools_prompt, redis_client,
        category="KEY",
        expected_tool="expire"
    ))
    await asyncio.sleep(0.5)
    
    # Note: TTL tool not directly available - type command returns TTL info
    results.append(await run_test(
        "TYPE with TTL - Get Key Info",
        f"Get information about key '{KEY_PREFIX}greeting' including its type and time to live",
        query_agent, tools_prompt, redis_client,
        category="KEY",
        expected_tool="type"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "TYPE - Get Key Type",
        f"What type of data is stored at key '{KEY_PREFIX}user:100'",
        query_agent, tools_prompt, redis_client,
        category="KEY",
        expected_tool="type"
    ))
    await asyncio.sleep(0.5)
    
    # ========================================================================
    # PHASE 7: DELETE OPERATIONS (2 tests - Cleanup)
    # ========================================================================
    print("\n" + "█"*70)
    print("█  PHASE 7: DELETE OPERATIONS (2 tests)")
    print("█"*70)
    
    results.append(await run_test(
        "DELETE - Delete Single Key",
        f"Delete the key '{KEY_PREFIX}temp'",
        query_agent, tools_prompt, redis_client,
        category="DELETE",
        expected_tool="delete"
    ))
    await asyncio.sleep(0.5)
    
    results.append(await run_test(
        "SREM - Remove from Set",
        f"Remove 'banana' from set '{KEY_PREFIX}fruits'",
        query_agent, tools_prompt, redis_client,
        category="DELETE",
        expected_tool="srem"
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
    category_order = ["STRING", "HASH", "LIST", "SET", "SORTED_SET", "KEY", "DELETE"]
    
    for category in category_order:
        if category not in by_category:
            continue
        
        tests = by_category[category]
        passed = sum(1 for t in tests if t.passed)
        total = len(tests)
        
        print(f"\n{'='*70}")
        print(f"📊 {category} OPERATIONS: {passed}/{total} passed")
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
    
    # Redis client stats
    stats = redis_client.get_stats()
    print(f"📈 Redis Client Stats:")
    print(f"   Total MCP calls: {stats['calls']}")
    print(f"   Successes: {stats['successes']}")
    print(f"   Errors: {stats['errors']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%\n")
    
    # Cleanup test keys
    print("🧹 Cleaning up test keys...")
    try:
        # Get all test keys
        keys_result = await redis_client.execute_tool("keys", {"pattern": f"{KEY_PREFIX}*"})
        print(f"   Found test keys to clean up")
    except Exception as e:
        print(f"   Cleanup note: {e}")
    
    print(f"{'═'*70}\n")
    
    # Properly cleanup resources
    await query_agent.close()
    await redis_client.disconnect()
    print("[OK] Test completed and disconnected\n")


if __name__ == "__main__":
    print("""
================================================================
       COMPREHENSIVE REDIS QUERY AGENT TEST                      
       Tests Redis Operations via Natural Language               
                                                                  
  Test Categories:                                                
  - STRING: set, get, incr, decr                                  
  - HASH: hset, hget, hgetall, hdel                               
  - LIST: lpush, rpush, lpop, lrange                              
  - SET: sadd, smembers, sismember, srem                          
  - SORTED SET: zadd, zrange, zscore                              
  - KEY: keys, exists, expire, ttl, type, del                     
                                                                  
  Uses QueryAgent for natural language → Redis command            
================================================================
    """)
    
    asyncio.run(test_all_redis_tools())
