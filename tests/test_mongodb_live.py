"""
Live MongoDB MCP Integration Test
==================================

This script tests the MongoDB MCP client by:
1. Connecting to MongoDB Atlas via MCP server
2. Listing available tools
3. Connecting to MongoDB instance
4. Inserting test data
5. Querying the data back

Configuration:
    Set MONGODB_CONNECTION_STRING environment variable or pass directly.
    
Usage:
    python tests/test_mongodb_live.py
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp.mongodb import MongoDBMCPClient, MongoDBToolManager


# MongoDB Atlas connection string
# Format: mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_CONNECTION_STRING = os.getenv(
    "MONGODB_CONNECTION_STRING",
    "mongodb+srv://faizan:Immalik%40123@temp.4z0ubzj.mongodb.net/?appName=temp"
)

# Test database and collection
TEST_DATABASE = "test_db"
TEST_COLLECTION = "mcp_test_collection"


async def test_mongodb_mcp():
    """Main test function"""
    print("=" * 60)
    print("MongoDB MCP Live Integration Test")
    print("=" * 60)
    
    # Create client
    print("\n📌 Step 1: Creating MongoDB MCP Client...")
    client = MongoDBMCPClient(
        connection_string=MONGODB_CONNECTION_STRING,
        timeout=60,
        startup_timeout=120
    )
    
    print(f"   Configured: {client.is_configured}")
    
    try:
        # Connect
        print("\n📌 Step 2: Connecting to MongoDB MCP Server...")
        print("   (This may take a minute to download and start the MCP server)")
        
        connected = await client.connect()
        
        if not connected:
            print("❌ Failed to connect to MongoDB MCP server")
            return False
        
        print(f"✅ Connected successfully!")
        
        # List tools
        print("\n📌 Step 3: Discovering available tools...")
        tools = await client.list_tools()
        
        print(f"   Found {len(tools)} tools:")
        for tool in tools[:10]:  # Show first 10
            print(f"   - {tool.name}: {tool.description[:60]}...")
        
        if len(tools) > 10:
            print(f"   ... and {len(tools) - 10} more tools")
        
        # Connect to MongoDB instance first
        print("\n📌 Step 4: Connecting to MongoDB instance...")
        connect_result = await client.execute_tool("connect", {
            "connectionString": MONGODB_CONNECTION_STRING
        })
        
        if connect_result.success:
            print(f"✅ Connected to MongoDB instance")
            print(f"   Result: {connect_result.result}")
        else:
            print(f"❌ Failed to connect: {connect_result.error}")
            return False
        
        # Insert test document
        print("\n📌 Step 5: Inserting test document...")
        
        test_document = {
            "name": "MCP Test User",
            "email": "mcp_test@example.com",
            "created_at": datetime.now().isoformat(),
            "test_id": f"test_{int(datetime.now().timestamp())}",
            "metadata": {
                "source": "mongodb_mcp_test",
                "version": "1.0"
            }
        }
        
        print(f"   Document to insert: {json.dumps(test_document, indent=2)}")
        
        # Use insert-one (kebab-case as per MCP server)
        result = await client.execute_tool("insert-one", {
            "database": TEST_DATABASE,
            "collection": TEST_COLLECTION,
            "document": test_document
        })
        
        print(f"\n   Insert result:")
        print(f"   Success: {result.success}")
        if result.success:
            print(f"   Result: {result.result}")
        else:
            print(f"   Error: {result.error}")
        
        # Query the data back
        print("\n📌 Step 6: Querying inserted data...")
        
        result = await client.execute_tool("find", {
            "database": TEST_DATABASE,
            "collection": TEST_COLLECTION,
            "filter": {"metadata.source": "mongodb_mcp_test"}
        })
        
        print(f"\n   Query result:")
        print(f"   Success: {result.success}")
        if result.success:
            print(f"   Data: {result.result}")
        else:
            print(f"   Error: {result.error}")
        
        # List collections
        print("\n📌 Step 7: List collections...")
        result = await client.execute_tool("list-collections", {
            "database": TEST_DATABASE
        })
        
        print(f"   Success: {result.success}")
        if result.success:
            print(f"   Collections: {result.result}")
        else:
            print(f"   Error: {result.error}")
        
        # Get stats
        print("\n📌 Step 8: Client Statistics")
        stats = client.get_stats()
        print(f"   Connected: {stats['connected']}")
        print(f"   Tools available: {stats['tools_available']}")
        print(f"   Calls: {stats['calls']}")
        print(f"   Successes: {stats['successes']}")
        print(f"   Errors: {stats['errors']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Disconnect
        print("\n📌 Disconnecting...")
        await client.disconnect()
        print("✅ Disconnected")


async def test_with_tool_manager():
    """Test using MongoDBToolManager (higher-level API)"""
    print("\n" + "=" * 60)
    print("Testing MongoDBToolManager")
    print("=" * 60)
    
    async with MongoDBToolManager(
        connection_string=MONGODB_CONNECTION_STRING,
        prefix="mongodb_",
        timeout=60
    ) as manager:
        
        if not manager.is_initialized:
            print("❌ Manager failed to initialize")
            return False
        
        print(f"✅ Manager initialized with {len(manager.get_tool_names())} tools")
        print(f"   Tools: {manager.get_tool_names()[:5]}...")
        
        # First connect to MongoDB instance
        connect_result = await manager.execute("mongodb_connect", {
            "connectionString": MONGODB_CONNECTION_STRING
        })
        print(f"\n   Connect result: {connect_result['success']}")
        
        # Insert via manager (using kebab-case tool name)
        result = await manager.execute("mongodb_insert-one", {
            "database": TEST_DATABASE,
            "collection": TEST_COLLECTION,
            "document": {
                "name": "Manager Test",
                "timestamp": datetime.now().isoformat(),
                "via": "MongoDBToolManager"
            }
        })
        
        print(f"\n   Insert via manager:")
        print(f"   Success: {result['success']}")
        print(f"   Provider: {result['provider']}")
        if result['success']:
            print(f"   Result: {result['result']}")
        else:
            print(f"   Error: {result['error']}")
        
        return result['success']


if __name__ == "__main__":
    print(f"\nMongoDB Connection: {MONGODB_CONNECTION_STRING[:50]}...")
    print(f"Test Database: {TEST_DATABASE}")
    print(f"Test Collection: {TEST_COLLECTION}")
    
    # Run tests
    success = asyncio.run(test_mongodb_mcp())
    
    if success:
        # Also test the tool manager
        asyncio.run(test_with_tool_manager())
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
