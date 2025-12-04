"""
MCP (Model Context Protocol) Integration Module
================================================

Provides integration with MCP servers for external service access:
- Zapier MCP: 8000+ application integrations  
- MongoDB MCP: MongoDB Atlas database operations

Main Components:
    - MCPClient: Base MCP client for protocol communication
    - ZapierMCPClient: Zapier-specific high-level client
    - ZapierToolManager: Bridge to existing tool_manager pattern
    - MongoDBMCPClient: MongoDB Atlas MCP client
    - MongoDBToolManager: MongoDB tool manager for LLM integration
    - MCPSecurityManager: Secure credential management

Quick Start (Zapier):
    from core.mcp import ZapierToolManager, MCPSecurityManager
    
    security = MCPSecurityManager()
    zapier_tools = ZapierToolManager(security)
    await zapier_tools.initialize()
    result = await zapier_tools.execute(
        tool_name="zapier_gmail_send_email",
        params={"to": "user@example.com", "subject": "Hi", "body": "Hello"}
    )

Quick Start (MongoDB):
    from core.mcp import MongoDBToolManager
    
    mongodb = MongoDBToolManager(connection_string="mongodb+srv://...")
    await mongodb.initialize()
    result = await mongodb.execute(
        tool_name="mongodb_find",
        params={"database": "mydb", "collection": "users", "filter": {}}
    )

For full documentation, see core/mcp/IMPLEMENTATION_PLAN.md
"""

# Exceptions
from .exceptions import (
    MCPError,
    MCPAuthenticationError,
    MCPConnectionError,
    MCPToolExecutionError,
    MCPRateLimitError,
    MCPValidationError,
    MCPServerError
)

# Security
from .security import (
    MCPSecurityManager,
    MCPCredentials
)

# Transport
from .transport import (
    MCPTransport,
    StreamableHTTPTransport,
    StdioTransport,
    MCPRequest,
    MCPResponse,
    MCPMethod,
    RateLimiter,
    ConnectionPool,
    JSONRPCErrorCode
)

# Client
from .client import (
    MCPClient,
    MCPTool,
    MCPToolResult
)

# Zapier Integration
from .zapier_integration import (
    ZapierMCPClient,
    ZapierToolManager,
    ZapierTool,
    ZapierToolCategory,
    get_zapier_tools_prompt
)

# MongoDB Integration
from .mongodb import (
    MongoDBMCPClient,
    MongoDBToolManager,
    MongoDBTool,
    MongoDBToolResult
)

# Query Agent (Universal NoSQL)
from .query_agent import (
    QueryAgent,
    QueryResult,
    LLMConfig
)

__all__ = [
    # Exceptions
    "MCPError",
    "MCPAuthenticationError",
    "MCPConnectionError",
    "MCPToolExecutionError",
    "MCPRateLimitError",
    "MCPValidationError",
    "MCPServerError",
    
    # Security
    "MCPSecurityManager",
    "MCPCredentials",
    
    # Transport
    "MCPTransport",
    "StreamableHTTPTransport",
    "StdioTransport",
    "MCPRequest",
    "MCPResponse",
    "MCPMethod",
    "RateLimiter",
    "ConnectionPool",
    "JSONRPCErrorCode",
    
    # Client
    "MCPClient",
    "MCPTool",
    "MCPToolResult",
    
    # Zapier
    "ZapierMCPClient",
    "ZapierToolManager",
    "ZapierTool",
    "ZapierToolCategory",
    "get_zapier_tools_prompt",
    
    # MongoDB
    "MongoDBMCPClient",
    "MongoDBToolManager",
    "MongoDBTool",
    "MongoDBToolResult",
    
    # Query Agent
    "QueryAgent",
    "QueryResult",
    "LLMConfig"
]

# Version
__version__ = "1.2.0"
