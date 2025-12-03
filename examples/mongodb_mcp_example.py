"""
MongoDB MCP Usage Example
=========================

This example shows how to use the MongoDB MCP client to interact
with MongoDB Atlas via the official MCP server.

Prerequisites:
1. Node.js installed (for npx)
2. MongoDB connection string

Run:
    python examples/mongodb_mcp_example.py
"""

import asyncio
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp import MongoDBMCPClient, MongoDBToolManager


async def example_with_client():
    """Example using MongoDBMCPClient directly"""
    print("=" * 60)
    print("Example 1: Using MongoDBMCPClient directly")
    print("=" * 60)
    
    # Create client with connection string
    # You can also set MONGODB_CONNECTION_STRING env var
    connection_string = os.getenv(
        "MONGODB_CONNECTION_STRING",
        "mongodb+srv://user:password@cluster.mongodb.net/"
    )
    
    client = MongoDBMCPClient(
        connection_string=connection_string,
        timeout=30,
        startup_timeout=120
    )
    
    try:
        # Connect to MongoDB MCP server
        print("\n🔗 Connecting to MongoDB MCP server...")
        connected = await client.connect()
        
        if not connected:
            print("❌ Failed to connect")
            return
        
        print("✅ Connected!")
        
        # List available tools
        print("\n📋 Available tools:")
        tools = await client.list_tools()
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")
        
        # Execute a tool (list databases)
        print("\n🚀 Executing listDatabases...")
        result = await client.execute_tool("listDatabases", {})
        
        if result.success:
            print(f"✅ Success! Databases: {result.result}")
        else:
            print(f"❌ Error: {result.error}")
        
        # Get stats
        print("\n📊 Statistics:")
        stats = client.get_stats()
        print(f"  Calls: {stats['calls']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        
    finally:
        await client.disconnect()
        print("\n✅ Disconnected")


async def example_with_tool_manager():
    """Example using MongoDBToolManager (recommended for LLM integration)"""
    print("\n" + "=" * 60)
    print("Example 2: Using MongoDBToolManager (LLM integration)")
    print("=" * 60)
    
    connection_string = os.getenv(
        "MONGODB_CONNECTION_STRING",
        "mongodb+srv://user:password@cluster.mongodb.net/"
    )
    
    # Use async context manager
    async with MongoDBToolManager(
        connection_string=connection_string,
        prefix="mongodb_"
    ) as manager:
        
        if not manager.is_initialized:
            print("❌ Failed to initialize")
            return
        
        print("✅ Initialized!")
        
        # Get tool schemas (for LLM function calling)
        print("\n📋 Tool schemas for LLM:")
        schemas = manager.get_tool_schemas()
        for name, schema in list(schemas.items())[:3]:
            print(f"  - {name}")
            print(f"    Description: {schema['description'][:50]}...")
        
        # Get prompt for LLM
        print("\n📝 LLM prompt snippet:")
        prompt = manager.get_tools_prompt()
        print(prompt[:500] + "...")
        
        # Execute a tool (simulating LLM calling)
        print("\n🚀 Executing mongodb_find...")
        result = await manager.execute(
            "mongodb_find",
            {
                "database": "test",
                "collection": "users",
                "filter": {}
            }
        )
        
        print(f"  Provider: {result['provider']}")
        print(f"  Success: {result['success']}")
        if result['error']:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Result: {str(result['result'])[:100]}...")


async def main():
    """Run examples"""
    print("\n🚀 MongoDB MCP Examples")
    print("=" * 60)
    print()
    print("This example demonstrates the MongoDB MCP integration.")
    print("Make sure to set MONGODB_CONNECTION_STRING environment variable.")
    print()
    
    # Check if connection string is configured
    if not os.getenv("MONGODB_CONNECTION_STRING"):
        print("⚠️  MONGODB_CONNECTION_STRING not set!")
        print("   Set it in your .env file or environment:")
        print("   export MONGODB_CONNECTION_STRING='mongodb+srv://...'")
        print()
        print("   Skipping actual connection test.")
        return
    
    # Run examples
    await example_with_client()
    await example_with_tool_manager()


if __name__ == "__main__":
    asyncio.run(main())
