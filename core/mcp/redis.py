"""
Redis MCP Integration
=======================

MCP client for Redis Database - connects to Redis's official MCP server.
Same pattern as MongoDB MCP integration.

Redis MCP Server: redis-mcp-server (PyPI)
Docs: https://github.com/redis/mcp-redis

The MCP server provides tools for:
- String operations (set, get, mset, mget, setex, setnx, getdel, etc.)
- Hash operations (hset, hget, hmset, hmget, hgetall, hdel, hkeys, etc.)
- List operations (lpush, rpush, lpop, rpop, lrange, llen, etc.)
- Set operations (sadd, srem, smembers, sismember, sinter, sunion, etc.)
- Sorted set operations (zadd, zrem, zrange, zscore, zrank, etc.)
- Key operations (keys, exists, del, expire, ttl, type, etc.)
- Pub/Sub (publish, subscribe)
- Streams (xadd, xread, xrange, xlen, etc.)
- JSON operations (json.set, json.get, json.del, etc.)
- Search/Query engine operations

Usage:
    from core.mcp.redis import RedisMCPClient, RedisToolManager
    
    # Using RedisMCPClient directly
    client = RedisMCPClient(redis_url="redis://...")
    await client.connect()
    tools = await client.list_tools()
    result = await client.execute_tool("set", {"key": "foo", "value": "bar"})
    
    # Or using RedisToolManager (recommended - like MongoDBToolManager)
    manager = RedisToolManager(redis_url="redis://...")
    await manager.initialize()
    tools = manager.get_tool_schemas()
    result = await manager.execute("redis_set", {"key": "foo", "value": "bar"})
"""

import asyncio
import logging
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .security import MCPSecurityManager
from .transport import StdioTransport, MCPRequest, MCPResponse, MCPMethod
from .exceptions import (
    MCPError,
    MCPConnectionError,
    MCPToolExecutionError,
)

logger = logging.getLogger(__name__)


@dataclass
class RedisTool:
    """Represents a tool from Redis MCP server"""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def required_params(self) -> List[str]:
        """Get required parameters from schema"""
        return self.input_schema.get("required", [])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
            "required": self.required_params
        }


@dataclass
class RedisToolResult:
    """Result of Redis tool execution"""
    success: bool
    result: Any = None
    error: str = None
    execution_time_ms: float = 0
    tool_name: str = ""


class RedisMCPClient:
    """
    Redis MCP Client - connects to Redis's official MCP server.
    
    The Redis MCP server runs as a subprocess and communicates via stdio.
    This is the same pattern as MongoDB MCP server.
    
    MCP Server: uvx --from redis-mcp-server@latest redis-mcp-server --url <redis_url>
    
    The server exposes tools for:
    - String operations: set, get, mset, mget, setex, setnx, getdel, append, incr, decr
    - Hash operations: hset, hget, hmset, hmget, hgetall, hdel, hkeys, hvals, hexists
    - List operations: lpush, rpush, lpop, rpop, lrange, llen, lindex, lset
    - Set operations: sadd, srem, smembers, sismember, sinter, sunion, sdiff, scard
    - Sorted set operations: zadd, zrem, zrange, zrevrange, zscore, zrank, zcard
    - Key operations: keys, exists, del, expire, ttl, type, rename, persist
    - Pub/Sub: publish, subscribe
    - Streams: xadd, xread, xrange, xlen, xinfo
    - JSON: json.set, json.get, json.del, json.arrappend
    - Search: ft.search, ft.create, ft.info
    """
    
    # UVX command to run Redis MCP server
    UVX_COMMAND = "uvx"
    MCP_SERVER_PACKAGE = "redis-mcp-server@latest"
    MCP_SERVER_CMD = "redis-mcp-server"
    
    def __init__(
        self,
        redis_url: str = None,
        timeout: int = 30,
        startup_timeout: int = 60,  # Redis MCP server startup time
        security_manager: Optional[MCPSecurityManager] = None,
    ):
        """
        Initialize Redis MCP client.
        
        Args:
            redis_url: Redis connection URL (redis://...)
            timeout: Request timeout in seconds
            startup_timeout: Timeout for server startup in seconds
            security_manager: Optional security manager for credential handling
        """
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.security_manager = security_manager
        
        # Get Redis URL from param or environment (.env)
        self._redis_url = redis_url or os.getenv("REDIS_MCP_URL", "")
        
        self._transport: Optional[StdioTransport] = None
        self._connected = False
        self._tools: Dict[str, RedisTool] = {}
        self._server_info: Dict[str, Any] = {}
        
        # Stats
        self._call_count = 0
        self._success_count = 0
        self._error_count = 0
        self._connect_time: Optional[datetime] = None
        
        logger.info("✅ RedisMCPClient initialized")
        if self._redis_url:
            masked = self._mask_redis_url(self._redis_url)
            logger.info(f"   Redis URL: {masked}")
    
    @staticmethod
    def _mask_redis_url(url: str) -> str:
        """Mask password in Redis URL for logging"""
        # Mask password in redis://user:password@host:port format
        return re.sub(r':([^@/:]+)@', r':****@', url)
    
    @property
    def is_configured(self) -> bool:
        """Check if Redis MCP is configured"""
        return bool(self._redis_url)
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to MCP server"""
        return self._connected and self._transport is not None and self._transport.is_connected
    
    async def connect(self) -> bool:
        """
        Connect to Redis MCP server.
        
        Starts the MCP server as a subprocess and connects via stdio.
        
        Returns:
            True if connection successful
        """
        if self.is_connected:
            logger.debug("Already connected to Redis MCP")
            return True
        
        if not self.is_configured:
            logger.error("❌ Redis not configured. Set redis_url or REDIS_MCP_URL env var")
            raise MCPConnectionError("Redis not configured - no Redis URL provided")
        
        try:
            logger.info("🔗 Starting Redis MCP server...")
            
            # Create stdio transport
            # The Redis MCP server expects --url argument for connection
            self._transport = StdioTransport(
                command=self.UVX_COMMAND,
                args=["--from", self.MCP_SERVER_PACKAGE, self.MCP_SERVER_CMD, "--url", self._redis_url],
                env={},  # Redis MCP uses --url arg, not env var
                timeout=self.timeout,
                startup_timeout=self.startup_timeout
            )
            
            # Connect (this starts the process and sends initialize)
            self._connected = await self._transport.connect()
            
            if self._connected:
                self._connect_time = datetime.now(timezone.utc)
                
                # Load available tools
                await self._discover_tools()
                
                logger.info(f"✅ Connected to Redis MCP server")
                logger.info(f"   Available tools: {len(self._tools)}")
                
                if self._tools:
                    logger.info(f"   Tools: {', '.join(list(self._tools.keys())[:5])}...")
            else:
                logger.error("❌ Failed to connect to Redis MCP server")
            
            return self._connected
            
        except FileNotFoundError:
            logger.error(f"❌ uvx not found. Make sure uv/uvx is installed (pip install uv)")
            raise MCPConnectionError("uvx not found - install uv package manager")
        except Exception as e:
            logger.error(f"❌ Redis MCP connection failed: {e}")
            await self.disconnect()
            raise MCPConnectionError(f"Connection failed: {e}")
    
    async def _discover_tools(self):
        """Discover available tools from MCP server"""
        if not self._transport:
            return
        
        try:
            # Send tools/list request
            request = MCPRequest(
                method=MCPMethod.TOOLS_LIST,
                params={}
            )
            
            response = await self._transport.send_request(request)
            
            if response.is_success and response.result:
                tools_data = response.result.get("tools", [])
                
                for tool_data in tools_data:
                    tool = RedisTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {})
                    )
                    self._tools[tool.name] = tool
                
                logger.info(f"📋 Discovered {len(self._tools)} Redis tools")
                
            elif response.error:
                logger.warning(f"⚠️ Could not discover tools: {response.error_message}")
                
        except Exception as e:
            logger.warning(f"⚠️ Tool discovery failed: {e}")
    
    async def disconnect(self):
        """Disconnect from Redis MCP server"""
        if self._transport:
            await self._transport.disconnect()
            self._transport = None
        
        self._connected = False
        self._tools.clear()
        
        logger.info("✅ Redis MCP disconnected")
    
    async def list_tools(self) -> List[RedisTool]:
        """
        Get list of available tools.
        
        Returns:
            List of RedisTool objects
        """
        if not self.is_connected:
            raise MCPConnectionError("Not connected to Redis MCP server")
        
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[RedisTool]:
        """Get specific tool by name"""
        return self._tools.get(name)
    
    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> RedisToolResult:
        """
        Execute a Redis MCP tool.
        
        Args:
            tool_name: Name of tool (set, get, hset, lpush, etc.)
            params: Tool parameters (key, value, field, etc.)
            
        Returns:
            RedisToolResult with execution result
        """
        if not self.is_connected:
            raise MCPConnectionError("Not connected to Redis MCP server")
        
        if not self._transport:
            raise MCPConnectionError("Transport not available")
        
        self._call_count += 1
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"🚀 Executing Redis tool: {tool_name}")
        logger.debug(f"   Params: {params}")
        
        try:
            # Send tools/call request
            request = MCPRequest(
                method=MCPMethod.TOOLS_CALL,
                params={
                    "name": tool_name,
                    "arguments": params
                }
            )
            
            response = await self._transport.send_request(request)
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if response.is_success:
                # Extract result from MCP response
                result_data = response.result
                
                # MCP tools/call returns content array
                if isinstance(result_data, dict) and "content" in result_data:
                    content = result_data.get("content", [])
                    if content and isinstance(content, list):
                        # Get text content
                        text_content = next(
                            (c.get("text") for c in content if c.get("type") == "text"),
                            str(content)
                        )
                        result_data = text_content
                
                # Check if the result text indicates an actual failure
                result_str = str(result_data).lower()
                failure_indicators = [
                    "connection refused",
                    "connection failed",
                    "failed to connect",
                    "error:",
                    "unable to",
                    "access denied",
                    "authentication failed",
                    "noauth",
                    "wrongpass"
                ]
                
                is_actual_failure = any(indicator in result_str for indicator in failure_indicators)
                
                if is_actual_failure:
                    self._error_count += 1
                    logger.error(f"❌ {tool_name} returned error in content: {result_data[:100]}...")
                    return RedisToolResult(
                        success=False,
                        error=str(result_data),
                        execution_time_ms=execution_time,
                        tool_name=tool_name
                    )
                
                self._success_count += 1
                logger.info(f"✅ {tool_name} completed in {execution_time:.0f}ms")
                
                return RedisToolResult(
                    success=True,
                    result=result_data,
                    execution_time_ms=execution_time,
                    tool_name=tool_name
                )
            else:
                self._error_count += 1
                error_msg = response.error_message or "Unknown error"
                
                logger.error(f"❌ {tool_name} failed: {error_msg}")
                
                return RedisToolResult(
                    success=False,
                    error=error_msg,
                    execution_time_ms=execution_time,
                    tool_name=tool_name
                )
                
        except asyncio.TimeoutError:
            self._error_count += 1
            error_msg = f"Tool execution timeout ({self.timeout}s)"
            logger.error(f"❌ {tool_name} timeout")
            
            return RedisToolResult(
                success=False,
                error=error_msg,
                tool_name=tool_name
            )
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"❌ Error executing {tool_name}: {e}")
            
            return RedisToolResult(
                success=False,
                error=str(e),
                tool_name=tool_name
            )
    
    def get_tools_prompt(self) -> str:
        """
        Generate tools description for LLM prompts.
        
        Same format as MongoDB tools - includes tool names,
        descriptions, and parameter info so LLM can generate correct queries.
        
        Returns:
            Formatted string describing available Redis tools
        """
        if not self._tools:
            return ""
        
        lines = [
            "## Redis Database Tools",
            "",
            "Available Redis operations (use exact tool names):",
            ""
        ]
        
        for name, tool in self._tools.items():
            desc = tool.description or name
            
            # Get parameter info from schema
            schema = tool.input_schema or {}
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Format parameters
            params_parts = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "any")
                is_req = "*" if param_name in required else ""
                params_parts.append(f"{param_name}{is_req}:{param_type}")
            
            params_str = ", ".join(params_parts) if params_parts else "no params"
            
            lines.append(f"- **{name}**: {desc}")
            lines.append(f"  Params: {params_str}")
            lines.append("")
        
        lines.append("NOTES:")
        lines.append("- * = required parameter")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "connected": self.is_connected,
            "connection_time": self._connect_time.isoformat() if self._connect_time else None,
            "tools_available": len(self._tools),
            "tool_names": list(self._tools.keys()),
            "calls": self._call_count,
            "successes": self._success_count,
            "errors": self._error_count,
            "success_rate": (self._success_count / max(self._call_count, 1)) * 100,
            "transport_stats": self._transport.get_stats() if self._transport else None
        }


class RedisToolManager:
    """
    Bridge between RedisMCPClient and OptimizedAgent.
    
    Same pattern as MongoDBToolManager - provides tool schemas
    for LLM function calling and routes execution to MCP.
    
    Usage:
        manager = RedisToolManager(
            security_manager=security_manager,
            redis_url="redis://..."
        )
        await manager.initialize()
        
        # Get tools for LLM
        schemas = manager.get_tool_schemas()
        
        # Execute tool (called by agent)
        result = await manager.execute("redis_set", {
            "key": "user:123",
            "value": "John Doe"
        })
    """
    
    def __init__(
        self,
        security_manager: Optional[MCPSecurityManager] = None,
        redis_url: str = None,
        prefix: str = "redis_",
        timeout: int = 30
    ):
        """
        Initialize Redis Tool Manager.
        
        Args:
            security_manager: Optional security manager
            redis_url: Redis connection URL
            prefix: Prefix for tool names (default: "redis_")
            timeout: Request timeout in seconds
        """
        self.security_manager = security_manager
        self.redis_url = redis_url
        self.prefix = prefix
        self.timeout = timeout
        
        self._client: Optional[RedisMCPClient] = None
        self._initialized = False
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """
        Initialize Redis MCP connection.
        
        Returns:
            True if initialization successful
        """
        try:
            self._client = RedisMCPClient(
                redis_url=self.redis_url,
                timeout=self.timeout,
                security_manager=self.security_manager
            )
            
            if not self._client.is_configured:
                logger.warning("⚠️ Redis MCP not configured - no Redis URL")
                return False
            
            connected = await self._client.connect()
            
            if connected:
                # Build tool schemas with prefix
                tools = await self._client.list_tools()
                
                for tool in tools:
                    prefixed_name = f"{self.prefix}{tool.name}"
                    self._tool_schemas[prefixed_name] = {
                        "name": prefixed_name,
                        "description": f"[Redis] {tool.description}",
                        "parameters": tool.input_schema,
                        "required": tool.required_params
                    }
                
                self._initialized = True
                logger.info(f"✅ RedisToolManager initialized ({len(self._tool_schemas)} tools)")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized"""
        return self._initialized
    
    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return self._client is not None and self._client.is_connected
    
    def get_tool_names(self) -> List[str]:
        """Get list of available tool names (with prefix)"""
        return list(self._tool_schemas.keys())
    
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get tool schemas for LLM function calling.
        
        Returns:
            Dictionary of tool schemas
        """
        return self._tool_schemas.copy()
    
    def get_tools_prompt(self) -> str:
        """
        Generate tools description for LLM system prompt.
        
        Returns:
            Formatted tools description
        """
        if self._client:
            return self._client.get_tools_prompt()
        return ""
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a Redis tool.
        
        Args:
            tool_name: Tool name (with or without prefix)
            params: Tool parameters
            
        Returns:
            Result dictionary with success, result, error fields
        """
        if not self._initialized or not self._client:
            return {
                "success": False,
                "error": "Redis not initialized",
                "provider": "redis_mcp"
            }
        
        # Remove prefix if present
        actual_name = tool_name
        if tool_name.startswith(self.prefix):
            actual_name = tool_name[len(self.prefix):]
        
        try:
            result = await self._client.execute_tool(actual_name, params)
            
            return {
                "success": result.success,
                "tool": tool_name,
                "result": result.result,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
                "provider": "redis_mcp"
            }
            
        except MCPConnectionError as e:
            return {
                "success": False,
                "tool": tool_name,
                "error": f"Connection error: {e}",
                "provider": "redis_mcp"
            }
        except Exception as e:
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e),
                "provider": "redis_mcp"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        stats = {
            "initialized": self._initialized,
            "tools_available": len(self._tool_schemas),
            "tool_names": list(self._tool_schemas.keys()),
            "prefix": self.prefix
        }
        
        if self._client:
            stats["client_stats"] = self._client.get_stats()
        
        return stats
    
    async def close(self):
        """Close Redis connection"""
        if self._client:
            await self._client.disconnect()
            self._client = None
        
        self._initialized = False
        self._tool_schemas.clear()
        
        logger.info("✅ RedisToolManager closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
