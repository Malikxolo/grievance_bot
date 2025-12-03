"""
MCP Transport Layer
===================

Handles low-level communication with MCP servers using the
Streamable HTTP transport protocol.

MCP Protocol Overview:
    - Uses JSON-RPC 2.0 over HTTP
    - Supports SSE (Server-Sent Events) for streaming responses
    - Endpoints: tools/list, tools/call, resources/list, resources/read

Transport Types:
    - StreamableHTTPTransport: Standard HTTP with SSE support (Zapier uses this)
    - StdioTransport: For local MCP servers (not needed for Zapier)

Reference:
    - MCP Specification: https://spec.modelcontextprotocol.io/
    - Zapier MCP Docs: https://docs.zapier.com/mcp/
"""

import asyncio
import json
import logging
import time
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import aiohttp
from aiohttp import ClientTimeout, ClientError, ClientResponseError

logger = logging.getLogger(__name__)


class MCPMethod(Enum):
    """MCP JSON-RPC methods"""
    # Tool methods
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    
    # Resource methods
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    
    # Prompt methods (for templates)
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    
    # Connection methods
    INITIALIZE = "initialize"
    PING = "ping"


# JSON-RPC Error Codes
class JSONRPCErrorCode:
    """Standard JSON-RPC 2.0 error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom MCP error codes (server-defined range: -32000 to -32099)
    SERVER_ERROR = -32000
    RATE_LIMITED = -32001
    AUTHENTICATION_FAILED = -32002
    TOOL_NOT_FOUND = -32003
    TOOL_EXECUTION_FAILED = -32004


@dataclass
class MCPRequest:
    """
    MCP JSON-RPC request structure.
    
    Format:
    {
        "jsonrpc": "2.0",
        "id": "unique-request-id",
        "method": "tools/call",
        "params": {...}
    }
    """
    method: MCPMethod
    params: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.request_id:
            # Generate unique ID with timestamp and random component
            self.request_id = f"req_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC format"""
        return {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": self.method.value if isinstance(self.method, MCPMethod) else self.method,
            "params": self.params
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict())


@dataclass
class MCPResponse:
    """
    MCP JSON-RPC response structure.
    
    Success format:
    {
        "jsonrpc": "2.0",
        "id": "request-id",
        "result": {...}
    }
    
    Error format:
    {
        "jsonrpc": "2.0",
        "id": "request-id",
        "error": {
            "code": -32600,
            "message": "Error description",
            "data": {...}
        }
    }
    """
    request_id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    http_status: Optional[int] = None
    
    @property
    def is_success(self) -> bool:
        return self.error is None and self.result is not None
    
    @property
    def error_code(self) -> Optional[int]:
        if self.error:
            return self.error.get("code")
        return None
    
    @property
    def error_message(self) -> Optional[str]:
        if self.error:
            return self.error.get("message")
        return None
    
    @property
    def is_rate_limited(self) -> bool:
        return self.error_code == JSONRPCErrorCode.RATE_LIMITED or self.http_status == 429
    
    @property
    def is_auth_error(self) -> bool:
        return self.error_code == JSONRPCErrorCode.AUTHENTICATION_FAILED or self.http_status in [401, 403]
    
    @property
    def is_retriable(self) -> bool:
        """Check if error is retriable (server errors, rate limits)"""
        if self.http_status and self.http_status >= 500:
            return True
        if self.http_status == 429:
            return True
        if self.error_code in [JSONRPCErrorCode.SERVER_ERROR, JSONRPCErrorCode.RATE_LIMITED]:
            return True
        return False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], latency_ms: float = None, http_status: int = None) -> "MCPResponse":
        """Parse JSON-RPC response"""
        return cls(
            request_id=data.get("id", "unknown"),
            result=data.get("result"),
            error=data.get("error"),
            raw_response=data,
            latency_ms=latency_ms,
            http_status=http_status
        )
    
    @classmethod
    def from_error(cls, request_id: str, code: int, message: str, data: Any = None, http_status: int = None) -> "MCPResponse":
        """Create error response"""
        return cls(
            request_id=request_id,
            error={
                "code": code,
                "message": message,
                "data": data
            },
            http_status=http_status
        )


class MCPTransport(ABC):
    """
    Abstract base class for MCP transport implementations.
    
    Subclasses must implement:
        - connect(): Establish connection
        - disconnect(): Close connection
        - send_request(): Send request and get response
        - is_connected: Connection status property
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to MCP server.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server"""
        pass
    
    @abstractmethod
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """
        Send request and receive response.
        
        Args:
            request: MCP request to send
            
        Returns:
            MCP response from server
        """
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected"""
        pass


class StdioTransport(MCPTransport):
    """
    Stdio transport for local MCP servers.
    
    Spawns an MCP server as a subprocess and communicates via stdin/stdout
    using JSON-RPC 2.0 protocol. This is the standard transport for local
    MCP servers like MongoDB Atlas MCP Server.
    
    Features:
        - Subprocess management (spawn, monitor, terminate)
        - JSON-RPC over stdin/stdout
        - Automatic process restart on failure
        - Environment variable injection
        - Request/response correlation via ID
    
    Usage:
        transport = StdioTransport(
            command="npx",
            args=["-y", "@mongodb-js/mongodb-mcp-server"],
            env={"MDB_MCP_CONNECTION_STRING": "mongodb+srv://..."}
        )
        await transport.connect()
        response = await transport.send_request(request)
        await transport.disconnect()
    """
    
    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        startup_timeout: int = 60,
        cwd: Optional[str] = None
    ):
        """
        Initialize Stdio transport.
        
        Args:
            command: Command to run (e.g., "npx", "node", "python")
            args: Command arguments (e.g., ["-y", "@mongodb-js/mongodb-mcp-server"])
            env: Environment variables to pass to subprocess
            timeout: Request timeout in seconds
            startup_timeout: Timeout for process startup in seconds
            cwd: Working directory for subprocess
        """
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.cwd = cwd
        
        self._process: Optional[asyncio.subprocess.Process] = None
        self._connected = False
        self._request_count = 0
        self._error_count = 0
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Build full command for logging
        full_cmd = f"{command} {' '.join(args)}"
        logger.info(f"✅ StdioTransport initialized")
        logger.info(f"   Command: {full_cmd}")
        logger.info(f"   Timeout: {timeout}s, Startup timeout: {startup_timeout}s")
    
    async def connect(self) -> bool:
        """
        Start subprocess and establish stdio communication.
        
        Returns:
            True if connection successful
        """
        if self._process is not None and self._process.returncode is None:
            logger.debug("Process already running, reusing")
            return True
        
        try:
            # Prepare environment (merge with current env)
            import os
            import platform
            process_env = os.environ.copy()
            process_env.update(self.env)
            
            # On Windows, we need to use .cmd extension for npx/npm
            command = self.command
            if platform.system() == "Windows":
                if command in ["npx", "npm", "node"]:
                    # Try to find the .cmd version
                    import shutil
                    cmd_version = f"{command}.cmd"
                    if shutil.which(cmd_version):
                        command = cmd_version
                    elif not shutil.which(command):
                        # If plain command doesn't work, try full path
                        nodejs_path = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "nodejs")
                        cmd_path = os.path.join(nodejs_path, cmd_version)
                        if os.path.exists(cmd_path):
                            command = cmd_path
            
            # Build command list
            cmd = [command] + self.args
            
            logger.info(f"🚀 Starting MCP server subprocess...")
            logger.debug(f"   Command: {' '.join(cmd)}")
            
            # Start subprocess
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
                cwd=self.cwd
            )
            
            logger.info(f"   Process started with PID: {self._process.pid}")
            
            # Start stderr reader for debugging
            asyncio.create_task(self._read_stderr())
            
            # Start stdout reader task
            self._reader_task = asyncio.create_task(self._read_responses())
            
            # Send initialize request to confirm server is ready
            init_request = MCPRequest(
                method=MCPMethod.INITIALIZE,
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "CS-Agent-MCP-Client",
                        "version": "1.0.0"
                    }
                }
            )
            
            # Wait for initialization with timeout
            try:
                response = await asyncio.wait_for(
                    self.send_request(init_request),
                    timeout=self.startup_timeout
                )
                
                if response.is_success:
                    self._connected = True
                    logger.info(f"✅ MCP server initialized successfully")
                    if response.result:
                        server_info = response.result.get("serverInfo", {})
                        logger.info(f"   Server: {server_info.get('name', 'Unknown')} v{server_info.get('version', 'Unknown')}")
                    return True
                else:
                    logger.error(f"❌ MCP server initialization failed: {response.error_message}")
                    await self.disconnect()
                    return False
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ MCP server startup timeout ({self.startup_timeout}s)")
                await self.disconnect()
                return False
                
        except FileNotFoundError:
            logger.error(f"❌ Command not found: {self.command}")
            logger.error("   Make sure the command is installed and in PATH")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to start MCP server: {e}")
            await self.disconnect()
            return False
    
    async def _read_stderr(self):
        """Read stderr from subprocess for debugging"""
        if not self._process or not self._process.stderr:
            return
            
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                stderr_text = line.decode().strip()
                if stderr_text:
                    logger.debug(f"[MCP stderr] {stderr_text}")
        except Exception as e:
            logger.debug(f"Stderr reader ended: {e}")
    
    async def _read_responses(self):
        """Read responses from subprocess stdout"""
        if not self._process or not self._process.stdout:
            return
            
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    logger.warning("MCP server stdout closed")
                    break
                    
                try:
                    response_text = line.decode().strip()
                    if not response_text:
                        continue
                        
                    logger.debug(f"[MCP response] {response_text[:200]}...")
                    
                    data = json.loads(response_text)
                    request_id = data.get("id")
                    
                    if request_id and request_id in self._pending_requests:
                        future = self._pending_requests.pop(request_id)
                        if not future.done():
                            future.set_result(data)
                    else:
                        # Handle notifications (no id) or orphan responses
                        logger.debug(f"Received notification or orphan response: {data}")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse MCP response: {e}")
                    
        except asyncio.CancelledError:
            logger.debug("Response reader cancelled")
        except Exception as e:
            logger.error(f"Response reader error: {e}")
            self._connected = False
    
    async def disconnect(self) -> None:
        """Terminate subprocess and cleanup"""
        self._connected = False
        
        # Cancel reader task
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        
        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
        
        # Terminate process
        if self._process:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Process did not terminate gracefully, killing...")
                    self._process.kill()
                    await self._process.wait()
                logger.info(f"✅ MCP server process terminated")
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")
            self._process = None
    
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """
        Send request to MCP server via stdin and wait for response.
        
        Args:
            request: MCP request to send
            
        Returns:
            MCP response from server
        """
        if not self._process or self._process.returncode is not None:
            return MCPResponse.from_error(
                request.request_id,
                JSONRPCErrorCode.INTERNAL_ERROR,
                "MCP server process not running"
            )
        
        start_time = time.time()
        
        async with self._lock:
            try:
                # Create future for response
                future: asyncio.Future = asyncio.Future()
                self._pending_requests[request.request_id] = future
                
                # Send request via stdin
                request_json = request.to_json() + "\n"
                self._process.stdin.write(request_json.encode())
                await self._process.stdin.drain()
                
                self._request_count += 1
                logger.debug(f"Sent request {request.request_id}: {request.method}")
                
            except Exception as e:
                self._error_count += 1
                if request.request_id in self._pending_requests:
                    del self._pending_requests[request.request_id]
                return MCPResponse.from_error(
                    request.request_id,
                    JSONRPCErrorCode.INTERNAL_ERROR,
                    f"Failed to send request: {e}"
                )
        
        # Wait for response with timeout
        try:
            data = await asyncio.wait_for(future, timeout=self.timeout)
            latency_ms = (time.time() - start_time) * 1000
            
            return MCPResponse.from_dict(data, latency_ms=latency_ms)
            
        except asyncio.TimeoutError:
            self._error_count += 1
            if request.request_id in self._pending_requests:
                del self._pending_requests[request.request_id]
            return MCPResponse.from_error(
                request.request_id,
                JSONRPCErrorCode.INTERNAL_ERROR,
                f"Request timeout after {self.timeout}s"
            )
        except asyncio.CancelledError:
            if request.request_id in self._pending_requests:
                del self._pending_requests[request.request_id]
            raise
    
    @property
    def is_connected(self) -> bool:
        """Check if transport is connected"""
        return (
            self._connected and 
            self._process is not None and 
            self._process.returncode is None
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics"""
        return {
            "type": "stdio",
            "command": f"{self.command} {' '.join(self.args)}",
            "connected": self.is_connected,
            "process_pid": self._process.pid if self._process else None,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "pending_requests": len(self._pending_requests)
        }


class StreamableHTTPTransport(MCPTransport):
    """
    Streamable HTTP transport for MCP.
    
    Uses aiohttp for async HTTP communication with Zapier MCP servers.
    
    Features:
        - HTTP POST for all requests (JSON-RPC over HTTP)
        - Automatic retry with exponential backoff
        - Rate limit handling (429 responses)
        - Connection keep-alive
        - Request timeout handling
        - SSE support for streaming responses (future)
    """
    
    # Default headers for MCP requests
    # Note: Zapier MCP requires both application/json and text/event-stream in Accept header
    DEFAULT_HEADERS = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "CS-Agent-MCP-Client/1.0"
    }
    
    def __init__(
        self,
        server_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        rate_limiter: Optional["RateLimiter"] = None
    ):
        """
        Initialize HTTP transport.
        
        Args:
            server_url: MCP server URL (Zapier endpoint)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for retriable errors
            retry_delay: Initial delay between retries (exponential backoff)
            retry_max_delay: Maximum delay between retries
            rate_limiter: Optional rate limiter instance
        """
        self.server_url = server_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_max_delay = retry_max_delay
        self.rate_limiter = rate_limiter or RateLimiter()
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._request_count = 0
        self._error_count = 0
        
        # Mask URL for logging
        self._masked_url = self._mask_url(server_url)
        
        logger.info(f"✅ StreamableHTTPTransport initialized")
        logger.info(f"   Server: {self._masked_url}")
        logger.info(f"   Timeout: {timeout}s, Max retries: {max_retries}")
    
    @staticmethod
    def _mask_url(url: str) -> str:
        """Mask sensitive parts of URL for logging"""
        if not url:
            return "[NO_URL]"
        try:
            if "://" in url:
                parts = url.split("/")
                if len(parts) > 3:
                    # Keep scheme and host, mask path
                    return f"{parts[0]}//{parts[2]}/***MASKED***"
            return "***MASKED***"
        except Exception:
            return "***MASKED***"
    
    async def connect(self) -> bool:
        """
        Establish HTTP session and validate connection.
        
        Creates an aiohttp ClientSession and optionally sends an
        initialize request to verify the server is responding.
        
        Returns:
            True if connection successful
        """
        if self._session is not None and not self._session.closed:
            logger.debug("Session already exists, reusing")
            return True
        
        try:
            # Create timeout configuration
            timeout_config = ClientTimeout(
                total=self.timeout,
                connect=10,  # Connection timeout
                sock_read=self.timeout  # Read timeout
            )
            
            # Create session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=10,  # Max concurrent connections
                limit_per_host=5,  # Max per host
                keepalive_timeout=30,  # Keep connections alive
                enable_cleanup_closed=True
            )
            
            self._session = aiohttp.ClientSession(
                timeout=timeout_config,
                connector=connector,
                headers=self.DEFAULT_HEADERS.copy()
            )
            
            # Optionally verify connection with a ping (if supported)
            # For now, just mark as connected after session creation
            self._connected = True
            
            logger.info(f"✅ HTTP transport connected to {self._masked_url}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create HTTP session: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """
        Close HTTP session and clean up resources.
        """
        if self._session:
            try:
                await self._session.close()
                # Allow time for graceful shutdown
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            finally:
                self._session = None
        
        self._connected = False
        logger.info(f"✅ HTTP transport disconnected (requests: {self._request_count}, errors: {self._error_count})")
    
    def _parse_sse_response(self, text: str) -> Dict[str, Any]:
        """
        Parse Server-Sent Events (SSE) format response.
        
        Zapier MCP returns responses in SSE format:
            event: message
            data: {"result": {...}}
        
        Args:
            text: Raw SSE response text
            
        Returns:
            Parsed JSON data from the SSE response
            
        Raises:
            json.JSONDecodeError: If data cannot be parsed as JSON
        """
        # SSE format has lines like:
        # event: message
        # data: {"jsonrpc": "2.0", ...}
        # (empty line)
        
        data_lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('data:'):
                # Extract JSON after "data: "
                json_str = line[5:].strip()
                if json_str:
                    data_lines.append(json_str)
        
        if data_lines:
            # Join all data lines (for multi-line data)
            full_json = ''.join(data_lines)
            return json.loads(full_json)
        
        # If no data: prefix found, try parsing the whole text as JSON
        return json.loads(text)
    
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """
        Send HTTP request to MCP server with retry logic.
        
        Args:
            request: MCP request to send
            
        Returns:
            MCP response from server
        """
        if not self._session or self._session.closed:
            # Auto-reconnect
            connected = await self.connect()
            if not connected:
                return MCPResponse.from_error(
                    request_id=request.request_id,
                    code=JSONRPCErrorCode.SERVER_ERROR,
                    message="Failed to establish connection"
                )
        
        # Apply rate limiting
        await self.rate_limiter.wait_if_needed()
        
        last_error: Optional[Exception] = None
        last_response: Optional[MCPResponse] = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self._request_count += 1
                
                # Track timing
                start_time = time.time()
                
                # Send POST request
                logger.debug(f"📤 Sending request (attempt {attempt + 1}/{self.max_retries + 1}): {request.method}")
                
                async with self._session.post(
                    self.server_url,
                    json=request.to_dict()
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Log response status
                    logger.debug(f"📥 Response status: {response.status} ({latency_ms:.1f}ms)")
                    
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        self.rate_limiter.record_rate_limit(retry_after)
                        
                        if attempt < self.max_retries:
                            logger.warning(f"⚠️ Rate limited, waiting {retry_after}s before retry")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            return MCPResponse.from_error(
                                request_id=request.request_id,
                                code=JSONRPCErrorCode.RATE_LIMITED,
                                message=f"Rate limited after {self.max_retries} retries",
                                http_status=429
                            )
                    
                    # Handle server errors (retriable)
                    if response.status >= 500:
                        error_text = await response.text()
                        if attempt < self.max_retries:
                            delay = self._calculate_backoff(attempt)
                            logger.warning(f"⚠️ Server error {response.status}, retrying in {delay:.1f}s")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            return MCPResponse.from_error(
                                request_id=request.request_id,
                                code=JSONRPCErrorCode.SERVER_ERROR,
                                message=f"Server error: {response.status}",
                                data={"body": error_text[:500]},
                                http_status=response.status
                            )
                    
                    # Handle client errors (non-retriable)
                    if response.status >= 400:
                        error_text = await response.text()
                        self._error_count += 1
                        
                        error_code = JSONRPCErrorCode.SERVER_ERROR
                        if response.status in [401, 403]:
                            error_code = JSONRPCErrorCode.AUTHENTICATION_FAILED
                        elif response.status == 404:
                            error_code = JSONRPCErrorCode.METHOD_NOT_FOUND
                        
                        return MCPResponse.from_error(
                            request_id=request.request_id,
                            code=error_code,
                            message=f"HTTP error: {response.status}",
                            data={"body": error_text[:500]},
                            http_status=response.status
                        )
                    
                    # Parse response - handle both JSON and SSE formats
                    try:
                        content_type = response.headers.get("Content-Type", "")
                        text = await response.text()
                        
                        # Handle SSE (Server-Sent Events) format from Zapier
                        # Format: "event: message\ndata: {...json...}\n\n"
                        if "text/event-stream" in content_type or text.startswith("event:"):
                            data = self._parse_sse_response(text)
                        else:
                            # Standard JSON response
                            data = json.loads(text)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse response: {e}")
                        return MCPResponse.from_error(
                            request_id=request.request_id,
                            code=JSONRPCErrorCode.PARSE_ERROR,
                            message="Invalid response format",
                            data={"body": text[:500] if 'text' in dir() else str(e)},
                            http_status=response.status
                        )
                    
                    # Create and return response
                    mcp_response = MCPResponse.from_dict(
                        data,
                        latency_ms=latency_ms,
                        http_status=response.status
                    )
                    
                    # Record success for rate limiter
                    self.rate_limiter.record_success()
                    
                    if mcp_response.is_success:
                        logger.info(f"✅ Request successful: {request.method} ({latency_ms:.1f}ms)")
                    else:
                        self._error_count += 1
                        logger.warning(f"⚠️ Request returned error: {mcp_response.error_message}")
                    
                    return mcp_response
                    
            except asyncio.TimeoutError:
                # IMPORTANT: Timeout doesn't mean failure! The request may have succeeded.
                # For write operations (INSERT, UPDATE, DELETE, SEND), retrying could cause duplicates.
                # Therefore, we do NOT retry on timeout - safer to report the timeout and let user retry manually.
                last_error = asyncio.TimeoutError(f"Request timed out after {self.timeout}s")
                logger.warning(f"⚠️ Timeout occurred - NOT retrying (request may have succeeded)")
                # Don't retry on timeout - break out of retry loop
                break
                    
            except ClientError as e:
                last_error = e
                self._error_count += 1
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(f"⚠️ Client error: {e}, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                    
            except Exception as e:
                last_error = e
                self._error_count += 1
                logger.error(f"❌ Unexpected error: {e}")
                break
        
        # All retries exhausted
        error_message = str(last_error) if last_error else "Unknown error after retries"
        return MCPResponse.from_error(
            request_id=request.request_id,
            code=JSONRPCErrorCode.SERVER_ERROR,
            message=error_message
        )
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: delay * 2^attempt
        delay = self.retry_delay * (2 ** attempt)
        
        # Add jitter (±25%)
        jitter = delay * 0.25 * (random.random() * 2 - 1)
        delay += jitter
        
        # Cap at max delay
        return min(delay, self.retry_max_delay)
    
    @property
    def is_connected(self) -> bool:
        """Check connection status"""
        return self._connected and self._session is not None and not self._session.closed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics"""
        return {
            "connected": self.is_connected,
            "server": self._masked_url,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1) * 100,
            "rate_limiter": self.rate_limiter.get_stats()
        }


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for MCP requests.
    
    Prevents hitting Zapier rate limits by throttling requests.
    Supports adaptive rate limiting based on 429 responses.
    """
    requests_per_minute: int = 60
    requests_per_second: float = 2.0  # Soft limit for smoothing
    
    def __post_init__(self):
        self._tokens: float = float(self.requests_per_minute)
        self._last_refill: float = time.time()
        self._last_request: float = 0.0
        self._rate_limit_until: float = 0.0
        self._total_waits: int = 0
        self._total_wait_time: float = 0.0
    
    async def wait_if_needed(self) -> float:
        """
        Wait if rate limited or need to smooth requests.
        
        Returns:
            Seconds waited (0 if no wait needed)
        """
        waited = 0.0
        
        # Check if we're in a rate-limited period
        now = time.time()
        if now < self._rate_limit_until:
            wait_time = self._rate_limit_until - now
            logger.info(f"⏳ Rate limit cooldown: waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            waited += wait_time
            self._total_waits += 1
            self._total_wait_time += wait_time
        
        # Refill tokens
        self._refill_tokens()
        
        # Check if we need to wait for tokens
        if self._tokens < 1.0:
            # Calculate wait time for 1 token
            refill_rate = self.requests_per_minute / 60.0  # tokens per second
            wait_time = (1.0 - self._tokens) / refill_rate
            wait_time = min(wait_time, 60.0)  # Cap at 1 minute
            
            if wait_time > 0.1:  # Only wait if significant
                logger.debug(f"⏳ Smoothing requests: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                waited += wait_time
                self._total_waits += 1
                self._total_wait_time += wait_time
                self._refill_tokens()
        
        # Consume a token
        self._tokens = max(0, self._tokens - 1)
        self._last_request = time.time()
        
        return waited
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self._last_refill
        
        # Calculate tokens to add
        refill_rate = self.requests_per_minute / 60.0  # tokens per second
        tokens_to_add = elapsed * refill_rate
        
        # Add tokens up to max
        self._tokens = min(float(self.requests_per_minute), self._tokens + tokens_to_add)
        self._last_refill = now
    
    def record_rate_limit(self, retry_after: int = 60):
        """
        Record a rate limit response (429).
        
        Args:
            retry_after: Seconds to wait (from Retry-After header)
        """
        self._rate_limit_until = time.time() + retry_after
        # Reduce tokens to prevent further requests
        self._tokens = 0
        logger.warning(f"🚫 Rate limit recorded, cooldown until +{retry_after}s")
    
    def record_success(self):
        """Record a successful request (helps with adaptive limiting)"""
        # Could implement adaptive rate adjustment here
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            "tokens_available": round(self._tokens, 2),
            "tokens_max": self.requests_per_minute,
            "total_waits": self._total_waits,
            "total_wait_time": round(self._total_wait_time, 2),
            "in_cooldown": time.time() < self._rate_limit_until,
            "cooldown_remaining": max(0, self._rate_limit_until - time.time())
        }


class SSEStreamHandler:
    """
    Server-Sent Events handler for streaming MCP responses.
    
    Some MCP tools may return streaming responses via SSE.
    This handler processes the event stream.
    
    Usage:
        async for event in sse_handler.stream_events(response):
            process_event(event)
    """
    
    def __init__(self, response: aiohttp.ClientResponse):
        self.response = response
        self._buffer = ""
    
    async def stream_events(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield events from SSE stream.
        
        Yields:
            Parsed event data dictionaries
        """
        async for line in self.response.content:
            line = line.decode('utf-8').strip()
            
            if not line:
                # Empty line = end of event
                if self._buffer:
                    event = self._parse_event(self._buffer)
                    if event:
                        yield event
                    self._buffer = ""
                continue
            
            if line.startswith(':'):
                # Comment line, ignore
                continue
            
            self._buffer += line + "\n"
        
        # Handle final event if buffer not empty
        if self._buffer:
            event = self._parse_event(self._buffer)
            if event:
                yield event
    
    def _parse_event(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse SSE event from raw text"""
        event_type = "message"
        event_data = ""
        
        for line in raw.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                event_data += line[5:].strip()
            elif line.startswith("id:"):
                # Event ID, could be used for resumption
                pass
        
        if event_data:
            try:
                return {
                    "type": event_type,
                    "data": json.loads(event_data)
                }
            except json.JSONDecodeError:
                return {
                    "type": event_type,
                    "data": event_data
                }
        
        return None


# Connection pool for multiple concurrent requests
class ConnectionPool:
    """
    Connection pool for MCP transports.
    
    Manages multiple transport connections for concurrent requests
    with automatic health checking and reconnection.
    """
    
    def __init__(
        self, 
        server_url: str, 
        pool_size: int = 5,
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.server_url = server_url
        self.pool_size = pool_size
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._pool: List[StreamableHTTPTransport] = []
        self._available: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._initialized = False
        
        logger.info(f"✅ ConnectionPool created (size: {pool_size})")
    
    async def initialize(self):
        """Initialize the connection pool"""
        async with self._lock:
            if self._initialized:
                return
            
            # Create transports
            for i in range(self.pool_size):
                transport = StreamableHTTPTransport(
                    server_url=self.server_url,
                    timeout=self.timeout,
                    max_retries=self.max_retries
                )
                await transport.connect()
                self._pool.append(transport)
                await self._available.put(transport)
            
            self._initialized = True
            logger.info(f"✅ Connection pool initialized with {self.pool_size} connections")
    
    async def get_transport(self) -> StreamableHTTPTransport:
        """
        Get available transport from pool.
        
        Returns:
            An available transport instance
        """
        if not self._initialized:
            await self.initialize()
        
        transport = await self._available.get()
        
        # Check health and reconnect if needed
        if not transport.is_connected:
            await transport.connect()
        
        return transport
    
    async def release_transport(self, transport: StreamableHTTPTransport) -> None:
        """Return transport to pool"""
        await self._available.put(transport)
    
    async def close(self):
        """Close all connections in the pool"""
        async with self._lock:
            for transport in self._pool:
                await transport.disconnect()
            self._pool.clear()
            self._initialized = False
        
        logger.info("✅ Connection pool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "pool_size": self.pool_size,
            "available": self._available.qsize(),
            "in_use": self.pool_size - self._available.qsize(),
            "initialized": self._initialized
        }
