"""
Redis MCP Integration Tests
=============================

Tests for Redis MCP client that connects to Redis database
via the official redis-mcp-server.

Run: python -m pytest tests/test_redis_mcp.py -v

For live tests against actual Redis:
    python tests/test_redis_mcp.py --live
"""

import asyncio
import pytest
import os
import sys
import argparse
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.mcp.redis import (
    RedisMCPClient,
    RedisToolManager,
    RedisTool,
    RedisToolResult
)
from core.mcp.transport import StdioTransport, MCPResponse, MCPRequest
from core.mcp.exceptions import MCPConnectionError


class TestRedisTool:
    """Tests for RedisTool dataclass"""
    
    def test_tool_creation(self):
        """Test tool creation with all fields"""
        tool = RedisTool(
            name="set",
            description="Set the string value of a key",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "ex": {"type": "integer", "description": "Expire time in seconds"}
                },
                "required": ["key", "value"]
            }
        )
        
        assert tool.name == "set"
        assert "Set the string value" in tool.description
        assert tool.required_params == ["key", "value"]
    
    def test_tool_to_dict(self):
        """Test tool serialization"""
        tool = RedisTool(
            name="get",
            description="Get the value of a key"
        )
        
        data = tool.to_dict()
        assert data["name"] == "get"
        assert data["description"] == "Get the value of a key"
        assert "parameters" in data
        assert "required" in data
    
    def test_tool_required_params_empty(self):
        """Test required params when schema has no required field"""
        tool = RedisTool(name="ping")
        assert tool.required_params == []


class TestRedisToolResult:
    """Tests for RedisToolResult dataclass"""
    
    def test_success_result(self):
        """Test successful result"""
        result = RedisToolResult(
            success=True,
            result="OK",
            execution_time_ms=5.5,
            tool_name="set"
        )
        
        assert result.success is True
        assert result.result == "OK"
        assert result.error is None
        assert result.execution_time_ms == 5.5
    
    def test_error_result(self):
        """Test error result"""
        result = RedisToolResult(
            success=False,
            error="WRONGTYPE Operation against a key holding the wrong kind of value",
            tool_name="get"
        )
        
        assert result.success is False
        assert result.result is None
        assert "WRONGTYPE" in result.error


class TestRedisMCPClient:
    """Tests for RedisMCPClient"""
    
    def test_initialization(self):
        """Test client initialization"""
        client = RedisMCPClient(
            redis_url="redis://localhost:6379",
            timeout=60
        )
        
        assert client.is_configured is True
        assert client.timeout == 60
        assert client.is_connected is False
    
    def test_initialization_from_env(self):
        """Test initialization from environment variable"""
        with patch.dict(os.environ, {"REDIS_MCP_URL": "redis://localhost:6379"}):
            client = RedisMCPClient()
            assert client.is_configured is True
    
    def test_not_configured(self):
        """Test client without Redis URL"""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env var
            os.environ.pop("REDIS_MCP_URL", None)
            client = RedisMCPClient(redis_url=None)
            assert client.is_configured is False
    
    def test_mask_redis_url(self):
        """Test password masking in Redis URL"""
        url = "redis://default:secretpassword@redis.cloud.com:6379"
        masked = RedisMCPClient._mask_redis_url(url)
        
        assert "secretpassword" not in masked
        assert "****" in masked
        assert "default" in masked
    
    @pytest.mark.asyncio
    async def test_connect_not_configured(self):
        """Test connect fails when not configured"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("REDIS_MCP_URL", None)
            client = RedisMCPClient(redis_url=None)
            
            with pytest.raises(MCPConnectionError, match="not configured"):
                await client.connect()
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection with mocked transport"""
        client = RedisMCPClient(
            redis_url="redis://localhost:6379"
        )
        
        # Mock transport
        mock_transport = AsyncMock(spec=StdioTransport)
        mock_transport.connect = AsyncMock(return_value=True)
        mock_transport.is_connected = True
        mock_transport.send_request = AsyncMock(return_value=MCPResponse(
            request_id="test",
            result={"tools": [
                {"name": "set", "description": "Set key value", "inputSchema": {}},
                {"name": "get", "description": "Get key value", "inputSchema": {}}
            ]}
        ))
        mock_transport.disconnect = AsyncMock()
        mock_transport.get_stats = MagicMock(return_value={})
        
        with patch.object(client, '_transport', mock_transport):
            client._transport = mock_transport
            client._connected = True
            
            # Test tool discovery
            await client._discover_tools()
            
            assert len(client._tools) == 2
            assert "set" in client._tools
            assert "get" in client._tools
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution"""
        client = RedisMCPClient(
            redis_url="redis://localhost:6379"
        )
        
        # Setup mocked state
        mock_transport = AsyncMock()
        mock_transport.is_connected = True
        mock_transport.send_request = AsyncMock(return_value=MCPResponse(
            request_id="test",
            result={
                "content": [
                    {"type": "text", "text": "OK"}
                ]
            }
        ))
        
        client._transport = mock_transport
        client._connected = True
        client._tools = {"set": RedisTool(name="set")}
        
        result = await client.execute_tool("set", {"key": "test", "value": "hello"})
        
        assert result.success is True
        assert result.result == "OK"
        assert result.tool_name == "set"
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_connected(self):
        """Test tool execution fails when not connected"""
        client = RedisMCPClient(redis_url="redis://localhost:6379")
        
        with pytest.raises(MCPConnectionError, match="Not connected"):
            await client.execute_tool("get", {"key": "test"})
    
    def test_get_tools_prompt(self):
        """Test tools prompt generation"""
        client = RedisMCPClient(redis_url="redis://localhost:6379")
        client._tools = {
            "set": RedisTool(
                name="set",
                description="Set key value",
                input_schema={
                    "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                    "required": ["key", "value"]
                }
            ),
            "get": RedisTool(
                name="get",
                description="Get key value",
                input_schema={
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"]
                }
            )
        }
        
        prompt = client.get_tools_prompt()
        
        assert "Redis Database Tools" in prompt
        assert "set" in prompt
        assert "get" in prompt
        assert "key*" in prompt  # required param
    
    def test_get_stats(self):
        """Test statistics retrieval"""
        client = RedisMCPClient(redis_url="redis://localhost:6379")
        client._call_count = 10
        client._success_count = 8
        client._error_count = 2
        
        stats = client.get_stats()
        
        assert stats["calls"] == 10
        assert stats["successes"] == 8
        assert stats["errors"] == 2
        assert stats["success_rate"] == 80.0


class TestRedisToolManager:
    """Tests for RedisToolManager"""
    
    def test_initialization(self):
        """Test manager initialization"""
        manager = RedisToolManager(
            redis_url="redis://localhost:6379",
            prefix="redis_"
        )
        
        assert manager.prefix == "redis_"
        assert manager.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization with mocked client"""
        manager = RedisToolManager(redis_url="redis://localhost:6379")
        
        # Create a mock client
        mock_client = AsyncMock(spec=RedisMCPClient)
        mock_client.is_configured = True
        mock_client.is_connected = True
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.list_tools = AsyncMock(return_value=[
            RedisTool(name="set", description="Set value", input_schema={}),
            RedisTool(name="get", description="Get value", input_schema={})
        ])
        mock_client.get_tools_prompt = MagicMock(return_value="## Redis Tools\n...")
        
        with patch('core.mcp.redis.RedisMCPClient', return_value=mock_client):
            result = await manager.initialize()
            
            assert result is True
            assert manager.is_initialized is True
    
    def test_get_tool_names(self):
        """Test getting tool names with prefix"""
        manager = RedisToolManager(redis_url="redis://localhost:6379")
        manager._initialized = True
        manager._tool_schemas = {
            "redis_set": {"name": "redis_set"},
            "redis_get": {"name": "redis_get"}
        }
        
        names = manager.get_tool_names()
        
        assert "redis_set" in names
        assert "redis_get" in names
    
    @pytest.mark.asyncio
    async def test_execute_not_initialized(self):
        """Test execute fails when not initialized"""
        manager = RedisToolManager(redis_url="redis://localhost:6379")
        
        result = await manager.execute("redis_set", {"key": "test", "value": "hello"})
        
        assert result["success"] is False
        assert "not initialized" in result["error"]


# ============================================================================
# LIVE TESTS - Run against actual Redis server
# ============================================================================

class TestRedisMCPLive:
    """
    Live integration tests for Redis MCP.
    
    These tests run against an actual Redis server and require:
    1. REDIS_MCP_URL environment variable set
    2. uvx installed (pip install uv)
    3. Network access to Redis server
    
    Run with: python tests/test_redis_mcp.py --live
    """
    
    # Redis URL for testing (from user's provided URL)
    REDIS_TEST_URL = "redis://default:goe9pNoOtVvDYcg6gLaC4uTxAT57t32o@redis-14300.c264.ap-south-1-1.ec2.cloud.redislabs.com:14300"
    
    @pytest.fixture
    async def redis_client(self):
        """Create and connect Redis MCP client"""
        client = RedisMCPClient(
            redis_url=self.REDIS_TEST_URL,
            timeout=30,
            startup_timeout=60
        )
        
        try:
            connected = await client.connect()
            if not connected:
                pytest.skip("Could not connect to Redis MCP server")
            yield client
        finally:
            await client.disconnect()
    
    @pytest.mark.asyncio
    @pytest.mark.live
    async def test_live_connection(self, redis_client):
        """Test live connection to Redis MCP server"""
        assert redis_client.is_connected is True
        
        tools = await redis_client.list_tools()
        assert len(tools) > 0
        
        print(f"\n✅ Connected to Redis MCP")
        print(f"   Available tools: {len(tools)}")
        print(f"   Tool names: {[t.name for t in tools[:10]]}...")
    
    @pytest.mark.asyncio
    @pytest.mark.live
    async def test_live_set_get(self, redis_client):
        """Test SET and GET operations"""
        test_key = "test:mcp:hello"
        test_value = "world"
        
        # SET
        set_result = await redis_client.execute_tool("set", {
            "key": test_key,
            "value": test_value
        })
        
        print(f"\n📝 SET {test_key} = {test_value}")
        print(f"   Result: {set_result}")
        
        assert set_result.success is True
        
        # GET
        get_result = await redis_client.execute_tool("get", {
            "key": test_key
        })
        
        print(f"\n📖 GET {test_key}")
        print(f"   Result: {get_result}")
        
        assert get_result.success is True
        assert test_value in str(get_result.result)
    
    @pytest.mark.asyncio
    @pytest.mark.live
    async def test_live_hash_operations(self, redis_client):
        """Test HASH operations (hset, hget, hgetall)"""
        hash_key = "test:mcp:user:123"
        
        # HSET
        hset_result = await redis_client.execute_tool("hset", {
            "key": hash_key,
            "field": "name",
            "value": "John Doe"
        })
        
        print(f"\n📝 HSET {hash_key} name = John Doe")
        print(f"   Result: {hset_result}")
        
        # HGET
        hget_result = await redis_client.execute_tool("hget", {
            "key": hash_key,
            "field": "name"
        })
        
        print(f"\n📖 HGET {hash_key} name")
        print(f"   Result: {hget_result}")
        
        assert hget_result.success is True
    
    @pytest.mark.asyncio
    @pytest.mark.live
    async def test_live_list_operations(self, redis_client):
        """Test LIST operations (lpush, rpush, lrange)"""
        list_key = "test:mcp:queue"
        
        # Clear the list first
        await redis_client.execute_tool("del", {"key": list_key})
        
        # LPUSH
        lpush_result = await redis_client.execute_tool("lpush", {
            "key": list_key,
            "element": "first"
        })
        
        print(f"\n📝 LPUSH {list_key} first")
        print(f"   Result: {lpush_result}")
        
        # RPUSH
        rpush_result = await redis_client.execute_tool("rpush", {
            "key": list_key,
            "element": "last"
        })
        
        print(f"\n📝 RPUSH {list_key} last")
        print(f"   Result: {rpush_result}")
        
        # LRANGE
        lrange_result = await redis_client.execute_tool("lrange", {
            "key": list_key,
            "start": 0,
            "stop": -1
        })
        
        print(f"\n📖 LRANGE {list_key} 0 -1")
        print(f"   Result: {lrange_result}")
        
        assert lrange_result.success is True
    
    @pytest.mark.asyncio
    @pytest.mark.live
    async def test_live_tools_prompt(self, redis_client):
        """Test tools prompt generation"""
        prompt = redis_client.get_tools_prompt()
        
        print(f"\n📋 Tools Prompt:\n{prompt[:500]}...")
        
        assert "Redis Database Tools" in prompt
        assert len(prompt) > 100
    
    @pytest.mark.asyncio
    @pytest.mark.live
    async def test_live_stats(self, redis_client):
        """Test statistics after operations"""
        # Do a simple operation
        await redis_client.execute_tool("set", {
            "key": "test:stats",
            "value": "check"
        })
        
        stats = redis_client.get_stats()
        
        print(f"\n📊 Client Stats:")
        print(f"   Connected: {stats['connected']}")
        print(f"   Tools: {stats['tools_available']}")
        print(f"   Calls: {stats['calls']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        
        assert stats['connected'] is True
        assert stats['calls'] > 0


class TestRedisMCPWithQueryAgent:
    """
    Tests for Redis MCP with QueryAgent (natural language queries).
    
    These tests demonstrate the universal QueryAgent design where
    natural language instructions are converted to Redis commands.
    """
    
    REDIS_TEST_URL = "redis://default:goe9pNoOtVvDYcg6gLaC4uTxAT57t32o@redis-14300.c264.ap-south-1-1.ec2.cloud.redislabs.com:14300"
    
    @pytest.fixture
    async def setup_with_query_agent(self):
        """Setup Redis client and QueryAgent"""
        from core.mcp.query_agent import QueryAgent
        from core.llm_client import LLMClient
        from core.config import Config
        
        # Create LLM client for QueryAgent
        config = Config()
        brain_config = config.get_brain_llm_config()
        llm_client = LLMClient(brain_config)
        
        # Create Redis client
        redis_client = RedisMCPClient(
            redis_url=self.REDIS_TEST_URL,
            timeout=30,
            startup_timeout=60
        )
        
        # Create QueryAgent
        query_agent = QueryAgent(llm_client=llm_client)
        
        try:
            connected = await redis_client.connect()
            if not connected:
                pytest.skip("Could not connect to Redis MCP server")
            
            yield {
                "redis_client": redis_client,
                "query_agent": query_agent
            }
        finally:
            await redis_client.disconnect()
    
    @pytest.mark.asyncio
    @pytest.mark.live
    @pytest.mark.slow
    async def test_natural_language_set(self, setup_with_query_agent):
        """Test natural language SET operation"""
        redis_client = setup_with_query_agent["redis_client"]
        query_agent = setup_with_query_agent["query_agent"]
        
        # Get tools prompt
        tools_prompt = redis_client.get_tools_prompt()
        
        # Natural language instruction
        instruction = "Store the value 'premium_user' for the key 'user:456:status'"
        
        print(f"\n🗣️ Natural Language Query: {instruction}")
        
        result = await query_agent.execute(
            tools_prompt=tools_prompt,
            instruction=instruction,
            mcp_client=redis_client
        )
        
        print(f"   Tool Called: {result.tool_name}")
        print(f"   Params: {result.params}")
        print(f"   Success: {result.success}")
        print(f"   Result: {result.result}")
        
        assert result.success is True or result.needs_clarification is True
    
    @pytest.mark.asyncio
    @pytest.mark.live
    @pytest.mark.slow
    async def test_natural_language_get(self, setup_with_query_agent):
        """Test natural language GET operation"""
        redis_client = setup_with_query_agent["redis_client"]
        query_agent = setup_with_query_agent["query_agent"]
        
        # First, set a value
        await redis_client.execute_tool("set", {
            "key": "user:789:name",
            "value": "Alice Smith"
        })
        
        # Get tools prompt
        tools_prompt = redis_client.get_tools_prompt()
        
        # Natural language instruction
        instruction = "Get the value stored at key 'user:789:name'"
        
        print(f"\n🗣️ Natural Language Query: {instruction}")
        
        result = await query_agent.execute(
            tools_prompt=tools_prompt,
            instruction=instruction,
            mcp_client=redis_client
        )
        
        print(f"   Tool Called: {result.tool_name}")
        print(f"   Params: {result.params}")
        print(f"   Success: {result.success}")
        print(f"   Result: {result.result}")
        
        if result.success:
            assert "Alice" in str(result.result) or result.tool_name == "get"
    
    @pytest.mark.asyncio
    @pytest.mark.live
    @pytest.mark.slow
    async def test_natural_language_hash(self, setup_with_query_agent):
        """Test natural language HASH operation"""
        redis_client = setup_with_query_agent["redis_client"]
        query_agent = setup_with_query_agent["query_agent"]
        
        # Get tools prompt
        tools_prompt = redis_client.get_tools_prompt()
        
        # Natural language instruction
        instruction = "Store user data in a hash: key 'customer:100', field 'email', value 'test@example.com'"
        
        print(f"\n🗣️ Natural Language Query: {instruction}")
        
        result = await query_agent.execute(
            tools_prompt=tools_prompt,
            instruction=instruction,
            mcp_client=redis_client
        )
        
        print(f"   Tool Called: {result.tool_name}")
        print(f"   Params: {result.params}")
        print(f"   Success: {result.success}")
        print(f"   Result: {result.result}")
    
    @pytest.mark.asyncio
    @pytest.mark.live
    @pytest.mark.slow
    async def test_natural_language_list(self, setup_with_query_agent):
        """Test natural language LIST operation"""
        redis_client = setup_with_query_agent["redis_client"]
        query_agent = setup_with_query_agent["query_agent"]
        
        # Get tools prompt
        tools_prompt = redis_client.get_tools_prompt()
        
        # Natural language instruction
        instruction = "Add 'task:complete_report' to the beginning of the list 'job_queue'"
        
        print(f"\n🗣️ Natural Language Query: {instruction}")
        
        result = await query_agent.execute(
            tools_prompt=tools_prompt,
            instruction=instruction,
            mcp_client=redis_client
        )
        
        print(f"   Tool Called: {result.tool_name}")
        print(f"   Params: {result.params}")
        print(f"   Success: {result.success}")
        print(f"   Result: {result.result}")


def run_live_tests():
    """Run live tests against actual Redis server"""
    print("\n" + "="*60)
    print("🔴 Redis MCP Live Integration Tests")
    print("="*60)
    print(f"\nUsing Redis URL: redis://default:****@redis-14300...redislabs.com:14300")
    print("\nRunning tests...\n")
    
    # Run pytest with live marker
    pytest.main([
        __file__,
        "-v",
        "-m", "live",
        "--tb=short",
        "-s"  # Show print output
    ])


def run_unit_tests():
    """Run unit tests only (no live connection needed)"""
    print("\n" + "="*60)
    print("🧪 Redis MCP Unit Tests")
    print("="*60)
    
    pytest.main([
        __file__,
        "-v",
        "-m", "not live",
        "--tb=short"
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redis MCP Tests")
    parser.add_argument("--live", action="store_true", help="Run live tests against actual Redis server")
    parser.add_argument("--all", action="store_true", help="Run all tests including live")
    args = parser.parse_args()
    
    if args.live:
        run_live_tests()
    elif args.all:
        run_unit_tests()
        run_live_tests()
    else:
        run_unit_tests()
