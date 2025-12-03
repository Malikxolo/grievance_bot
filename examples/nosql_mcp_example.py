"""
NoSQL MCP Example Usage
=======================

This script demonstrates how to use the NoSQL MCP module to:
1. Connect to MongoDB (or other NoSQL databases)
2. Execute natural language queries
3. Run structured queries

Run:
    python examples/nosql_mcp_example.py
    
Or with environment variable:
    MONGODB_URI="mongodb+srv://..." python examples/nosql_mcp_example.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp.nosql import (
    NoSQLMCPManager,
    DatabaseType,
    NoSQLDatabaseConfig,
    MongoDBAdapter
)


async def main():
    """Main example function"""
    
    print("=" * 60)
    print("NoSQL MCP Module - Usage Example")
    print("=" * 60)
    
    # Get MongoDB connection string from environment or use default
    mongodb_uri = os.environ.get(
        "MONGODB_URI",
        # Default: User's provided MongoDB Atlas connection
        "mongodb+srv://faizan:Immalik@123@temp.4z0ubzj.mongodb.net/?appName=temp"
    )
    
    # ========== SETUP ==========
    print("\n📦 Setting up NoSQL MCP Manager...")
    
    # Initialize manager
    manager = NoSQLMCPManager()
    
    # Register the MongoDB adapter
    manager.register_adapter(DatabaseType.MONGODB, MongoDBAdapter)
    
    # Register database configuration
    manager.register_database(NoSQLDatabaseConfig(
        name="mongodb_main",
        db_type=DatabaseType.MONGODB,
        connection_string=mongodb_uri,
        database="test_db",
        auto_connect=True
    ))
    
    print("✅ Manager configured with MongoDB adapter")
    
    # ========== CONNECT & INITIALIZE ==========
    async with manager:
        print("\n🔌 Connected to MongoDB!")
        
        # List available databases
        databases = manager.list_databases()
        print(f"\n📊 Available databases: {databases}")
        
        # ========== EXAMPLE 1: Natural Language Query ==========
        print("\n" + "=" * 60)
        print("EXAMPLE 1: Natural Language Query")
        print("=" * 60)
        
        nl_query = "find all documents from users collection"
        print(f"\n🗣️  Natural Language: '{nl_query}'")
        
        result = await manager.execute_query(
            natural_query=nl_query,
            database="mongodb_main",
            collection="users"
        )
        
        print(f"✅ Success: {result['success']}")
        print(f"📋 Documents found: {result['count']}")
        print(f"⏱️  Execution time: {result['execution_time_ms']}ms")
        if result['result']:
            print(f"📄 Sample result: {result['result'][:2]}")
        
        # ========== EXAMPLE 2: Structured Query ==========
        print("\n" + "=" * 60)
        print("EXAMPLE 2: Structured Query (Insert)")
        print("=" * 60)
        
        # Insert test document
        insert_result = await manager.execute_structured_query(
            {
                "operation": "insert",
                "collection": "example_collection",
                "documents": [
                    {"name": "Alice", "age": 30, "city": "New York"},
                    {"name": "Bob", "age": 25, "city": "Los Angeles"}
                ]
            },
            database="mongodb_main"
        )
        
        print(f"✅ Inserted: {insert_result['result']}")
        
        # ========== EXAMPLE 3: Query with Filter ==========
        print("\n" + "=" * 60)
        print("EXAMPLE 3: Natural Language with Filter")
        print("=" * 60)
        
        nl_query = "find users with age > 25"
        print(f"\n🗣️  Natural Language: '{nl_query}'")
        
        result = await manager.execute_query(
            natural_query=nl_query,
            database="mongodb_main",
            collection="example_collection"
        )
        
        print(f"✅ Success: {result['success']}")
        print(f"📋 Documents matching age > 25: {result['count']}")
        if result['result']:
            print(f"📄 Results: {result['result']}")
        
        # ========== EXAMPLE 4: Count ==========
        print("\n" + "=" * 60)
        print("EXAMPLE 4: Count Query")
        print("=" * 60)
        
        count_result = await manager.execute_structured_query(
            {
                "operation": "count",
                "collection": "example_collection",
                "filter": {}
            },
            database="mongodb_main"
        )
        
        print(f"📊 Total documents in collection: {count_result['count']}")
        
        # ========== CLEANUP ==========
        print("\n" + "=" * 60)
        print("CLEANUP")
        print("=" * 60)
        
        # Delete test documents
        cleanup_result = await manager.execute_structured_query(
            {
                "operation": "delete",
                "collection": "example_collection",
                "filter": {},
                "multi": True
            },
            database="mongodb_main"
        )
        
        print(f"🗑️  Cleaned up {cleanup_result['result']['deleted_count']} documents")
        
        # ========== HEALTH CHECK ==========
        print("\n" + "=" * 60)
        print("HEALTH CHECK")
        print("=" * 60)
        
        health = await manager.health_check()
        print(f"❤️  Health status: {health}")
    
    print("\n✅ All examples completed successfully!")
    print("=" * 60)


# ========== ADDING NEW DATABASES ==========
def show_how_to_add_new_database():
    """
    Shows how to add a new database.
    
    The NoSQL MCP module is designed to be UNIVERSAL.
    Adding a new database is as simple as:
    
    1. Create a new adapter (e.g., redis_adapter.py)
    2. Register it with the manager
    3. That's it!
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║         HOW TO ADD A NEW DATABASE (UNIVERSAL DESIGN)          ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║  1. Create adapter file: adapters/redis_adapter.py           ║
║                                                               ║
║     class RedisAdapter(BaseNoSQLAdapter):                     ║
║         async def connect(self): ...                          ║
║         async def execute(self, query): ...                   ║
║         async def health_check(self): ...                     ║
║         def get_supported_operations(self): ...               ║
║                                                               ║
║  2. Register in manager:                                      ║
║                                                               ║
║     manager.register_adapter(DatabaseType.REDIS, RedisAdapter)║
║                                                               ║
║  3. Add configuration:                                        ║
║                                                               ║
║     manager.register_database(NoSQLDatabaseConfig(            ║
║         name="redis_cache",                                   ║
║         db_type=DatabaseType.REDIS,                           ║
║         connection_string="redis://localhost:6379"            ║
║     ))                                                        ║
║                                                               ║
║  4. Use it:                                                   ║
║                                                               ║
║     result = await manager.execute_query(                     ║
║         "get user:123",                                       ║
║         database="redis_cache"                                ║
║     )                                                         ║
║                                                               ║
║  OR via environment variables:                                ║
║     REDIS_URL="redis://localhost:6379"                        ║
║                                                               ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    # Show how to add new databases
    show_how_to_add_new_database()
    
    # Run the example
    asyncio.run(main())
