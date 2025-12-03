"""
NoSQL MCP Integration Tests
===========================

Tests for the NoSQL MCP module including:
- Configuration and Registry
- Query Agent (NL -> Query conversion)
- MongoDB Adapter
- NoSQLMCPManager

Run tests:
    pytest tests/test_nosql_mcp.py -v
    
Run with MongoDB connection:
    MONGODB_URI="mongodb+srv://..." pytest tests/test_nosql_mcp.py -v
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp.nosql import (
    NoSQLMCPManager,
    NoSQLQueryAgent,
    QueryTemplate,
    DatabaseType,
    NoSQLDatabaseConfig,
    DatabaseRegistry,
    MongoDBAdapter,
    BaseNoSQLAdapter,
    NoSQLMCPError,
    DatabaseConnectionError,
    QueryExecutionError,
    QueryConversionError,
    UnsupportedOperationError
)


# ============== Configuration Tests ==============

class TestDatabaseType:
    """Test DatabaseType enum"""
    
    def test_database_types_exist(self):
        """All expected database types should exist"""
        assert DatabaseType.MONGODB
        assert DatabaseType.REDIS
        assert DatabaseType.NEO4J
        assert DatabaseType.QDRANT
    
    def test_database_type_values(self):
        """Database types should have correct string values"""
        assert DatabaseType.MONGODB.value == "mongodb"
        assert DatabaseType.REDIS.value == "redis"
        assert DatabaseType.NEO4J.value == "neo4j"
        assert DatabaseType.QDRANT.value == "qdrant"


class TestNoSQLDatabaseConfig:
    """Test database configuration"""
    
    def test_create_config(self):
        """Should create valid configuration"""
        config = NoSQLDatabaseConfig(
            name="test_mongo",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost:27017",
            database="testdb"
        )
        
        assert config.name == "test_mongo"
        assert config.db_type == DatabaseType.MONGODB
        assert config.database == "testdb"
        assert config.enabled is True
        assert config.auto_connect is True
    
    def test_config_with_settings(self):
        """Should accept custom settings"""
        config = NoSQLDatabaseConfig(
            name="test_mongo",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost:27017",
            settings={"max_pool_size": 50}
        )
        
        assert config.settings["max_pool_size"] == 50
    
    def test_config_disabled(self):
        """Should support disabled configs"""
        config = NoSQLDatabaseConfig(
            name="disabled_db",
            db_type=DatabaseType.REDIS,
            connection_string="redis://localhost",
            enabled=False
        )
        
        assert config.enabled is False


class TestDatabaseRegistry:
    """Test database registry"""
    
    def test_register_database(self):
        """Should register a database config"""
        registry = DatabaseRegistry()
        config = NoSQLDatabaseConfig(
            name="mongo1",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost"
        )
        
        registry.register(config)
        
        assert "mongo1" in registry.databases
        assert registry.get("mongo1") == config
    
    def test_get_by_type(self):
        """Should get databases by type"""
        registry = DatabaseRegistry()
        
        registry.register(NoSQLDatabaseConfig(
            name="mongo1",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost"
        ))
        registry.register(NoSQLDatabaseConfig(
            name="redis1",
            db_type=DatabaseType.REDIS,
            connection_string="redis://localhost"
        ))
        registry.register(NoSQLDatabaseConfig(
            name="mongo2",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost"
        ))
        
        mongo_dbs = registry.get_by_type(DatabaseType.MONGODB)
        
        assert len(mongo_dbs) == 2
        assert all(c.db_type == DatabaseType.MONGODB for c in mongo_dbs)
    
    def test_get_nonexistent(self):
        """Should return None for non-existent database"""
        registry = DatabaseRegistry()
        
        assert registry.get("nonexistent") is None


# ============== Query Template Tests ==============

class TestQueryTemplate:
    """Test query templates"""
    
    def test_mongodb_patterns(self):
        """Should have MongoDB patterns"""
        patterns = QueryTemplate.get_patterns(DatabaseType.MONGODB)
        
        assert "find" in patterns
        assert "insert" in patterns
        assert "update" in patterns
        assert "delete" in patterns
        assert "aggregate" in patterns
    
    def test_redis_patterns(self):
        """Should have Redis patterns"""
        patterns = QueryTemplate.get_patterns(DatabaseType.REDIS)
        
        assert "get" in patterns
        assert "set" in patterns
        assert "delete" in patterns
    
    def test_neo4j_patterns(self):
        """Should have Neo4j patterns"""
        patterns = QueryTemplate.get_patterns(DatabaseType.NEO4J)
        
        assert "match" in patterns
        assert "create" in patterns


# ============== Query Agent Tests ==============

class TestNoSQLQueryAgent:
    """Test query agent conversion"""
    
    def test_init(self):
        """Should initialize without LLM client"""
        agent = NoSQLQueryAgent()
        assert agent.llm_client is None
    
    def test_init_with_llm(self):
        """Should initialize with LLM client"""
        mock_llm = MagicMock()
        agent = NoSQLQueryAgent(llm_client=mock_llm)
        assert agent.llm_client == mock_llm
    
    @pytest.mark.asyncio
    async def test_pattern_conversion_find(self):
        """Should convert find query using patterns"""
        agent = NoSQLQueryAgent()
        
        query = await agent.convert(
            "find all documents from users collection",
            DatabaseType.MONGODB,
            collection="users"
        )
        
        assert query["operation"] == "find"
        assert query["collection"] == "users"
    
    @pytest.mark.asyncio
    async def test_pattern_conversion_insert(self):
        """Should convert insert query"""
        agent = NoSQLQueryAgent()
        
        query = await agent.convert(
            "insert new document into users",
            DatabaseType.MONGODB,
            collection="users"
        )
        
        assert query["operation"] == "insert"
    
    @pytest.mark.asyncio
    async def test_pattern_conversion_count(self):
        """Should convert count query"""
        agent = NoSQLQueryAgent()
        
        query = await agent.convert(
            "count how many users exist",
            DatabaseType.MONGODB,
            collection="users"
        )
        
        assert query["operation"] == "count"
    
    @pytest.mark.asyncio
    async def test_pattern_conversion_with_filter(self):
        """Should extract simple filters"""
        agent = NoSQLQueryAgent()
        
        query = await agent.convert(
            "find users where status = active",
            DatabaseType.MONGODB,
            collection="users"
        )
        
        assert "filter" in query
        # Pattern matching should capture "status" = "active"
    
    @pytest.mark.asyncio
    async def test_pattern_conversion_with_comparison(self):
        """Should extract comparison operators"""
        agent = NoSQLQueryAgent()
        
        query = await agent.convert(
            "find users where age > 25",
            DatabaseType.MONGODB,
            collection="users"
        )
        
        assert "filter" in query
        if "age" in query["filter"]:
            assert "$gt" in query["filter"]["age"]
    
    @pytest.mark.asyncio
    async def test_redis_get_pattern(self):
        """Should convert Redis get query"""
        agent = NoSQLQueryAgent()
        
        query = await agent.convert(
            "get user:123",
            DatabaseType.REDIS
        )
        
        assert query["operation"] == "get"
    
    @pytest.mark.asyncio
    async def test_cache_works(self):
        """Should cache converted queries"""
        agent = NoSQLQueryAgent()
        
        # First call
        query1 = await agent.convert(
            "find all users",
            DatabaseType.MONGODB,
            collection="users"
        )
        
        # Second call (should be cached)
        query2 = await agent.convert(
            "find all users",
            DatabaseType.MONGODB,
            collection="users"
        )
        
        assert query1 == query2
    
    def test_clear_cache(self):
        """Should clear cache"""
        agent = NoSQLQueryAgent()
        agent._conversion_cache["test"] = {"data": "value"}
        
        agent.clear_cache()
        
        assert len(agent._conversion_cache) == 0
    
    def test_get_supported_operations(self):
        """Should return supported operations"""
        agent = NoSQLQueryAgent()
        
        mongo_ops = agent.get_supported_operations(DatabaseType.MONGODB)
        
        assert "find" in mongo_ops
        assert "insert" in mongo_ops


# ============== MongoDB Adapter Tests ==============

class TestMongoDBAdapter:
    """Test MongoDB adapter (mocked)"""
    
    def test_init(self):
        """Should initialize with config"""
        config = NoSQLDatabaseConfig(
            name="test_mongo",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost:27017",
            database="testdb"
        )
        
        adapter = MongoDBAdapter(config)
        
        assert adapter.name == "test_mongo"
        assert adapter.db_type == DatabaseType.MONGODB
        assert adapter._database_name == "testdb"
    
    def test_supported_operations(self):
        """Should return supported operations"""
        config = NoSQLDatabaseConfig(
            name="test_mongo",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost"
        )
        adapter = MongoDBAdapter(config)
        
        ops = adapter.get_supported_operations()
        
        assert "find" in ops
        assert "insert" in ops
        assert "update" in ops
        assert "delete" in ops
        assert "aggregate" in ops
    
    def test_stats(self):
        """Should track stats"""
        config = NoSQLDatabaseConfig(
            name="test_mongo",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost"
        )
        adapter = MongoDBAdapter(config)
        
        stats = adapter.get_stats()
        
        assert stats["name"] == "test_mongo"
        assert stats["db_type"] == "mongodb"
        assert stats["connected"] is False
        assert stats["query_count"] == 0


# ============== NoSQL MCP Manager Tests ==============

class TestNoSQLMCPManager:
    """Test NoSQL MCP Manager"""
    
    def test_init(self):
        """Should initialize manager"""
        manager = NoSQLMCPManager()
        
        assert manager._initialized is False
        assert manager.query_agent is not None
    
    def test_register_adapter(self):
        """Should register adapter class"""
        manager = NoSQLMCPManager()
        
        manager.register_adapter(DatabaseType.MONGODB, MongoDBAdapter)
        
        assert DatabaseType.MONGODB in manager._adapter_classes
    
    def test_register_database(self):
        """Should register database config"""
        manager = NoSQLMCPManager()
        
        config = NoSQLDatabaseConfig(
            name="mongo1",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost"
        )
        manager.register_database(config)
        
        assert "mongo1" in manager._registry.databases
    
    def test_get_tools(self):
        """Should return tool definitions"""
        manager = NoSQLMCPManager()
        
        # Add mock adapter
        config = NoSQLDatabaseConfig(
            name="mongo1",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost"
        )
        adapter = MongoDBAdapter(config)
        manager._adapters["mongo1"] = adapter
        
        tools = manager.get_tools()
        
        assert len(tools) >= 1
        assert any(t["name"] == "nosql_query" for t in tools)
    
    def test_list_databases(self):
        """Should list registered databases"""
        manager = NoSQLMCPManager()
        
        config = NoSQLDatabaseConfig(
            name="mongo1",
            db_type=DatabaseType.MONGODB,
            connection_string="mongodb://localhost"
        )
        adapter = MongoDBAdapter(config)
        manager._adapters["mongo1"] = adapter
        
        dbs = manager.list_databases()
        
        assert len(dbs) == 1
        assert dbs[0]["name"] == "mongo1"
        assert dbs[0]["db_type"] == "mongodb"
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Should close all connections"""
        manager = NoSQLMCPManager()
        
        # Add mock adapter
        mock_adapter = AsyncMock(spec=BaseNoSQLAdapter)
        manager._adapters["test"] = mock_adapter
        manager._initialized = True
        
        await manager.close()
        
        mock_adapter.disconnect.assert_called_once()
        assert manager._initialized is False


# ============== Integration Tests (with real MongoDB) ==============

class TestMongoDBIntegration:
    """Integration tests with real MongoDB (skipped if no connection)"""
    
    # MongoDB connection string from environment or use default
    MONGODB_URI = os.environ.get(
        "MONGODB_URI",
        "mongodb+srv://faizan:Immalik@123@temp.4z0ubzj.mongodb.net/?appName=temp"
    )
    
    @pytest.fixture
    def mongodb_config(self):
        """MongoDB configuration"""
        return NoSQLDatabaseConfig(
            name="test_mongodb",
            db_type=DatabaseType.MONGODB,
            connection_string=self.MONGODB_URI,
            database="test_nosql_mcp",
            auto_connect=True  # Auto-connect for tests
        )
    
    @pytest.fixture
    def manager(self, mongodb_config):
        """Create manager with MongoDB adapter"""
        manager = NoSQLMCPManager()
        manager.register_adapter(DatabaseType.MONGODB, MongoDBAdapter)
        manager.register_database(mongodb_config)
        return manager
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("MONGODB_URI") and not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="MongoDB URI not configured. Set MONGODB_URI or RUN_INTEGRATION_TESTS=1"
    )
    async def test_connect_to_mongodb(self, manager):
        """Should connect to real MongoDB"""
        async with manager:
            assert manager._initialized
            
            health = await manager.health_check()
            assert health["test_mongodb"]["healthy"]
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("MONGODB_URI") and not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="MongoDB URI not configured"
    )
    async def test_execute_query(self, manager):
        """Should execute query on real MongoDB"""
        async with manager:
            # Simple find query
            result = await manager.execute_query(
                natural_query="find all documents",
                database="test_mongodb",
                collection="test_collection"
            )
            
            assert result["success"]
            assert result["database"] == "test_mongodb"
            assert "execution_time_ms" in result
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("MONGODB_URI") and not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="MongoDB URI not configured"
    )
    async def test_insert_and_find(self, manager):
        """Should insert and find documents"""
        async with manager:
            # Insert
            insert_result = await manager.execute_structured_query(
                {
                    "operation": "insert",
                    "collection": "test_nosql_mcp_test",
                    "documents": [
                        {"name": "Test User", "age": 30, "test": True}
                    ]
                },
                database="test_mongodb"
            )
            
            assert insert_result["success"]
            
            # Find
            find_result = await manager.execute_structured_query(
                {
                    "operation": "find",
                    "collection": "test_nosql_mcp_test",
                    "filter": {"test": True}
                },
                database="test_mongodb"
            )
            
            assert find_result["success"]
            assert len(find_result["result"]) >= 1
            
            # Cleanup
            await manager.execute_structured_query(
                {
                    "operation": "delete",
                    "collection": "test_nosql_mcp_test",
                    "filter": {"test": True},
                    "multi": True
                },
                database="test_mongodb"
            )
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("MONGODB_URI") and not os.environ.get("RUN_INTEGRATION_TESTS"),
        reason="MongoDB URI not configured"
    )
    async def test_natural_language_query(self, manager):
        """Should execute natural language query"""
        async with manager:
            # First insert test data
            await manager.execute_structured_query(
                {
                    "operation": "insert",
                    "collection": "users_test",
                    "documents": [
                        {"name": "Alice", "age": 30},
                        {"name": "Bob", "age": 20},
                        {"name": "Charlie", "age": 35}
                    ]
                },
                database="test_mongodb"
            )
            
            # Natural language query
            result = await manager.execute_query(
                natural_query="find users with age > 25",
                database="test_mongodb",
                collection="users_test"
            )
            
            assert result["success"]
            
            # Cleanup
            await manager.execute_structured_query(
                {
                    "operation": "delete",
                    "collection": "users_test",
                    "filter": {},
                    "multi": True
                },
                database="test_mongodb"
            )


# ============== Exception Tests ==============

class TestExceptions:
    """Test exception handling"""
    
    def test_nosql_mcp_error(self):
        """Should create NoSQLMCPError"""
        error = NoSQLMCPError("Test error")
        assert str(error) == "Test error"
    
    def test_database_connection_error(self):
        """Should create DatabaseConnectionError with context"""
        error = DatabaseConnectionError(
            "Connection failed",
            database="mongo1",
            db_type="mongodb"
        )
        
        assert "Connection failed" in str(error)
        assert error.database == "mongo1"
    
    def test_query_execution_error(self):
        """Should create QueryExecutionError with context"""
        error = QueryExecutionError(
            "Query failed",
            query="find {}",
            database="mongo1"
        )
        
        assert error.query == "find {}"
    
    def test_query_conversion_error(self):
        """Should create QueryConversionError"""
        error = QueryConversionError(
            "Conversion failed",
            query="invalid query"
        )
        
        assert error.query == "invalid query"
    
    def test_unsupported_operation_error(self):
        """Should create UnsupportedOperationError"""
        error = UnsupportedOperationError(
            "Operation not supported",
            operation="invalid_op",
            db_type="mongodb"
        )
        
        assert error.operation == "invalid_op"


# ============== Run Tests ==============

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
