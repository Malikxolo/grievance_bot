"""
Simple MongoDB Test - Add Sample Data
======================================

This test adds sample data to MongoDB and verifies it works.

Run:
    python tests/test_mongodb_sample_data.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp.nosql import (
    NoSQLMCPManager,
    DatabaseType,
    NoSQLDatabaseConfig,
    MongoDBAdapter
)

from urllib.parse import quote_plus

# Your MongoDB credentials (password URL-encoded because it has @)
password = quote_plus("Immalik@123")
MONGODB_URI = f"mongodb+srv://faizan:{password}@temp.4z0ubzj.mongodb.net/?appName=temp"


async def test_add_sample_data():
    """Add sample data to MongoDB"""
    
    print("=" * 60)
    print("MongoDB Sample Data Test")
    print("=" * 60)
    
    # Setup
    manager = NoSQLMCPManager()
    manager.register_adapter(DatabaseType.MONGODB, MongoDBAdapter)
    manager.register_database(NoSQLDatabaseConfig(
        name="test_db",
        db_type=DatabaseType.MONGODB,
        connection_string=MONGODB_URI,
        database="sample_database"
    ))
    
    async with manager:
        print("\n✅ Connected to MongoDB!")
        
        # ========== ADD SAMPLE USERS ==========
        print("\n📝 Adding sample users...")
        
        users_result = await manager.execute_structured_query(
            {
                "operation": "insert",
                "collection": "users",
                "documents": [
                    {"name": "Ali Khan", "email": "ali@example.com", "age": 28, "city": "Karachi"},
                    {"name": "Sara Ahmed", "email": "sara@example.com", "age": 32, "city": "Lahore"},
                    {"name": "Usman Malik", "email": "usman@example.com", "age": 25, "city": "Islamabad"},
                    {"name": "Fatima Noor", "email": "fatima@example.com", "age": 30, "city": "Karachi"},
                    {"name": "Hassan Raza", "email": "hassan@example.com", "age": 35, "city": "Peshawar"},
                ]
            },
            database="test_db"
        )
        print(f"   Inserted {len(users_result['result']['inserted_ids'])} users")
        
        # ========== ADD SAMPLE PRODUCTS ==========
        print("\n📝 Adding sample products...")
        
        products_result = await manager.execute_structured_query(
            {
                "operation": "insert",
                "collection": "products",
                "documents": [
                    {"name": "Laptop", "price": 85000, "category": "Electronics", "stock": 15},
                    {"name": "Mobile Phone", "price": 45000, "category": "Electronics", "stock": 50},
                    {"name": "Headphones", "price": 5000, "category": "Electronics", "stock": 100},
                    {"name": "Chair", "price": 12000, "category": "Furniture", "stock": 25},
                    {"name": "Desk", "price": 25000, "category": "Furniture", "stock": 10},
                ]
            },
            database="test_db"
        )
        print(f"   Inserted {len(products_result['result']['inserted_ids'])} products")
        
        # ========== ADD SAMPLE ORDERS ==========
        print("\n📝 Adding sample orders...")
        
        orders_result = await manager.execute_structured_query(
            {
                "operation": "insert",
                "collection": "orders",
                "documents": [
                    {"user": "Ali Khan", "product": "Laptop", "quantity": 1, "total": 85000, "status": "completed"},
                    {"user": "Sara Ahmed", "product": "Mobile Phone", "quantity": 2, "total": 90000, "status": "pending"},
                    {"user": "Usman Malik", "product": "Headphones", "quantity": 3, "total": 15000, "status": "completed"},
                    {"user": "Fatima Noor", "product": "Chair", "quantity": 4, "total": 48000, "status": "shipped"},
                    {"user": "Hassan Raza", "product": "Desk", "quantity": 1, "total": 25000, "status": "pending"},
                ]
            },
            database="test_db"
        )
        print(f"   Inserted {len(orders_result['result']['inserted_ids'])} orders")
        
        # ========== VERIFY DATA ==========
        print("\n" + "=" * 60)
        print("VERIFYING DATA")
        print("=" * 60)
        
        # Count users
        users_count = await manager.execute_structured_query(
            {"operation": "count", "collection": "users", "filter": {}},
            database="test_db"
        )
        print(f"\n👥 Total users: {users_count['count']}")
        
        # Count products
        products_count = await manager.execute_structured_query(
            {"operation": "count", "collection": "products", "filter": {}},
            database="test_db"
        )
        print(f"📦 Total products: {products_count['count']}")
        
        # Count orders
        orders_count = await manager.execute_structured_query(
            {"operation": "count", "collection": "orders", "filter": {}},
            database="test_db"
        )
        print(f"🛒 Total orders: {orders_count['count']}")
        
        # ========== SAMPLE QUERIES ==========
        print("\n" + "=" * 60)
        print("SAMPLE QUERIES")
        print("=" * 60)
        
        # Find users from Karachi
        karachi_users = await manager.execute_structured_query(
            {"operation": "find", "collection": "users", "filter": {"city": "Karachi"}},
            database="test_db"
        )
        print(f"\n🏙️ Users from Karachi: {[u['name'] for u in karachi_users['result']]}")
        
        # Find products under 20000
        cheap_products = await manager.execute_structured_query(
            {"operation": "find", "collection": "products", "filter": {"price": {"$lt": 20000}}},
            database="test_db"
        )
        print(f"💰 Products under 20,000: {[p['name'] for p in cheap_products['result']]}")
        
        # Find pending orders
        pending_orders = await manager.execute_structured_query(
            {"operation": "find", "collection": "orders", "filter": {"status": "pending"}},
            database="test_db"
        )
        print(f"⏳ Pending orders: {[o['user'] for o in pending_orders['result']]}")
        
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nData is now in your MongoDB Atlas database: 'sample_database'")
        print("Collections created: users, products, orders")


if __name__ == "__main__":
    asyncio.run(test_add_sample_data())
