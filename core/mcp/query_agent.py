"""
Universal NoSQL Query Agent
===========================

A universal agent that converts natural language instructions to database operations.
Works with any NoSQL database (MongoDB, Redis, Cassandra, etc.) by receiving tools prompt.

Architecture:
    OptimizedAgent detects DB intent
        → Calls QueryAgent with tools_prompt + instruction
            → QueryAgent uses LLM to select tool + generate params
                → If required params missing → Returns clarification request
                → If all params available → Executes via MCP client
                    → Returns result

UNIVERSAL DESIGN:
    - Receives tools_prompt (same format as Zapier tools in OptimizedAgent)
    - NO hardcoded tool names or database-specific logic
    - Works with ANY database that provides get_tools_prompt()
    - NEVER halluccinates missing parameters - asks for clarification instead

Usage:
    from core.mcp.query_agent import QueryAgent
    
    # Get tools prompt from any MCP client (MongoDB, Redis, etc.)
    tools_prompt = mongodb_client.get_tools_prompt()
    
    agent = QueryAgent()
    result = await agent.execute(
        tools_prompt=tools_prompt,
        instruction="Add apple to fruits collection in mydb database",
        mcp_client=mongodb_client
    )
    
    # Check if clarification needed
    if result.needs_clarification:
        print(f"Need more info: {result.clarification_message}")
    elif result.success:
        print(f"Result: {result.result}")
"""

import os
import json
import logging
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QueryResult:
    """Result from query execution"""
    success: bool
    tool_name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    llm_response: Optional[str] = None
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    missing_fields: Optional[List[str]] = None


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str
    model: str
    api_key: str
    max_tokens: int = 4096
    temperature: float = 0.1
    base_url: Optional[str] = None


# =============================================================================
# SYSTEM PROMPT - UNIVERSAL FOR ALL DATABASES
# =============================================================================

SYSTEM_PROMPT = """You are a database query expert. Your job is to convert natural language instructions into database operations.

You will receive:
1. A list of available database tools with descriptions and required/optional parameters
2. A natural language instruction from the user

Your task:
1. Read the available tools carefully - note which parameters are REQUIRED (marked with *)
2. Select the EXACT tool name that matches the user's intent
3. Check if ALL required parameters (marked with *) can be extracted from the instruction
4. If ANY required parameter is missing, ask for clarification
5. For simple insert operations, you can create document/value structure from item names

CRITICAL RULES:
1. Use the EXACT tool name as shown in the tools list
2. If ANY required parameter (*) is not provided in the instruction, ask for clarification
3. Only include parameters that are explicitly needed for the operation
4. For find/query operations, only include filters that are explicitly mentioned
5. For update operations, you MUST know what to update - if not specified, ask for clarification
6. For delete operations, you MUST know what to delete - if not specified, ask for clarification

PARAMETER FORMAT RULES:
- NEVER include null values - omit optional parameters if not needed
- String parameters must be strings, NOT arrays
- Number parameters must be numbers
- When in doubt about optional params, omit them entirely

RESPONSE FORMAT (JSON only, no markdown, no backticks):

If all required parameters are available:
{
    "tool": "exact-tool-name",
    "params": {
        "param1": "value1",
        "param2": "value2"
    }
}

If ANY required parameter is MISSING:
{
    "tool": null,
    "needs_clarification": true,
    "missing_fields": ["field1", "field2"],
    "clarification_message": "Please specify: [list missing required parameters]"
}

If instruction cannot be mapped to any tool:
{
    "tool": null,
    "error": "reason"
}
"""


# =============================================================================
# QUERY AGENT CLASS
# =============================================================================

class QueryAgent:
    """
    Universal NoSQL Query Agent.
    
    Converts natural language to database operations using any MCP client.
    Completely database-agnostic - just provide the tools_prompt.
    
    This is a TOOL, not a chatbot. It does not maintain conversation context.
    If required information is missing, it asks for clarification.
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Initialize Query Agent.
        
        Args:
            llm_config: LLM configuration. If None, loads from .env (HEART config)
            llm_client: Shared LLMClient instance (preferred)
        """
        # Prefer shared LLMClient if provided (same infra as OptimizedAgent)
        if llm_client is not None:
            self.llm_client = llm_client
            self.llm_config = getattr(llm_client, "config", None)
            self._owns_client = False  # Don't close shared client
        else:
            # Load simple config from .env if not provided
            if llm_config is None:
                self.llm_config = self._load_config_from_env()
            else:
                self.llm_config = llm_config
            # Create a dedicated client using the universal LLMClient wrapper
            self.llm_client = LLMClient(self.llm_config)
            self._owns_client = True  # We own this client, must close it
        
        logger.info("✅ QueryAgent initialized")
        if self.llm_config:
            logger.info(f"   Provider: {self.llm_config.provider}")
            logger.info(f"   Model: {self.llm_config.model}")
    
    async def close(self) -> None:
        """
        Close resources (LLM client session).
        
        Call this when done using the QueryAgent to prevent
        unclosed session warnings.
        """
        if self._owns_client and self.llm_client:
            await self.llm_client.close_session()
            logger.debug("QueryAgent LLM client session closed")
    
    def _load_config_from_env(self) -> LLMConfig:
        """Load LLM config from environment variables (HEART config)"""
        provider = os.getenv("HEART_LLM_PROVIDER", "openrouter")
        model = os.getenv("HEART_LLM_MODEL", "meta-llama/llama-4-maverick")
        api_key = os.getenv("HEART_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError("No API key found. Set HEART_LLM_API_KEY or OPENROUTER_API_KEY in .env")

        # Match Config.create_llm_config base_url behavior so LLMClient hits correct endpoint
        base_url: Optional[str] = None
        if provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
        elif provider == "deepseek":
            base_url = "https://api.deepseek.com/v1"
        elif provider == "sarvam":
            base_url = "https://api.sarvam.ai/v1"

        return LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            max_tokens=4096,
            temperature=0.1,  # Low temp for precise query generation
            base_url=base_url,
        )
    
    async def execute(
        self,
        tools_prompt: str,
        instruction: str,
        mcp_client: Any
    ) -> QueryResult:
        """
        Execute a natural language instruction using available tools.
        
        This is a TOOL - no context/history. If required info is missing,
        returns clarification request instead of hallucinating.
        
        Args:
            tools_prompt: Formatted tools prompt from MCP client's get_tools_prompt()
            instruction: Natural language instruction from user
            mcp_client: MCP client with execute_tool(name, params) method
            
        Returns:
            QueryResult with execution results or clarification request
        """
        logger.info(f"🚀 QueryAgent executing: {instruction[:50]}...")
        
        try:
            # Step 1: Build prompt
            user_prompt = self._build_user_prompt(tools_prompt, instruction)
            
            # Step 2: Call LLM to get tool selection and params
            llm_response = await self._call_llm(user_prompt)
            logger.debug(f"LLM Response: {llm_response}")
            
            # Step 3: Parse LLM response
            parsed = self._parse_response(llm_response)
            
            # Step 4: Check if clarification is needed
            if parsed.get("needs_clarification"):
                return QueryResult(
                    success=False,
                    needs_clarification=True,
                    clarification_message=parsed.get("clarification_message", "Please provide more information."),
                    missing_fields=parsed.get("missing_fields", []),
                    llm_response=llm_response
                )
            
            if parsed.get("error") or parsed.get("tool") is None:
                return QueryResult(
                    success=False,
                    error=parsed.get("error", "Could not determine tool to use"),
                    llm_response=llm_response
                )
            
            tool_name = parsed["tool"]
            params = parsed.get("params", {})
            
            # Step 4.5: Sanitize parameters (fix common LLM mistakes)
            params = self._sanitize_params(params, tools_prompt, tool_name)
            
            logger.info(f"📋 Selected tool: {tool_name}")
            logger.info(f"   Params: {json.dumps(params, indent=2)}")
            
            # Step 5: Execute via MCP client
            result = await mcp_client.execute_tool(tool_name, params)
            
            # Step 6: Return result
            if hasattr(result, 'success'):
                # MongoDBToolResult style
                return QueryResult(
                    success=result.success,
                    tool_name=tool_name,
                    params=params,
                    result=result.result if result.success else None,
                    error=result.error if not result.success else None,
                    llm_response=llm_response
                )
            else:
                # Generic result
                return QueryResult(
                    success=True,
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    llm_response=llm_response
                )
                
        except Exception as e:
            logger.error(f"❌ QueryAgent error: {e}")
            return QueryResult(
                success=False,
                error=str(e)
            )
    
    def _build_user_prompt(
        self,
        tools_prompt: str,
        instruction: str
    ) -> str:
        """Build the user prompt with tools and instruction"""
        
        # Build prompt - tools_prompt is already formatted from get_tools_prompt()
        prompt = f"""Available Tools:
{tools_prompt}

User Instruction: {instruction}

Analyze the instruction carefully. If any required parameter (marked with *) is NOT explicitly mentioned, you MUST ask for clarification. DO NOT assume or guess any values.

Respond with JSON only."""

        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM via the shared LLMClient (same stack as OptimizedAgent)."""
        if not hasattr(self, "llm_client") or self.llm_client is None:
            raise RuntimeError("LLM client not configured for QueryAgent")

        temperature = getattr(self.llm_config, "temperature", 0.1) if self.llm_config else 0.1
        max_tokens = getattr(self.llm_config, "max_tokens", 4096) if self.llm_config else 4096

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract tool and params.
        
        UNIVERSAL DESIGN: Works with any database MCP client by extracting
        JSON from various LLM response formats (raw, markdown, mixed).
        """
        import re
        
        if not response or not response.strip():
            return {
                "tool": None,
                "error": "Empty response from LLM"
            }
        
        response = response.strip()
        
        # Strategy 1: Direct JSON parse (cleanest case)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON from markdown code block
        # Handles: ```json {...} ``` or ``` {...} ```
        json_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find the outermost balanced JSON object
        # This handles cases where LLM returns text before/after JSON
        brace_start = response.find('{')
        if brace_start != -1:
            # Find matching closing brace using a simple brace counter
            depth = 0
            in_string = False
            escape_next = False
            
            for i, char in enumerate(response[brace_start:], brace_start):
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            # Found complete JSON object
                            json_str = response[brace_start:i + 1]
                            try:
                                return json.loads(json_str)
                            except json.JSONDecodeError:
                                break
        
        # Strategy 4: Try to fix common LLM JSON errors
        # Sometimes LLMs add trailing commas or leave incomplete
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            json_str = json_match.group(0)
            # Remove trailing commas before closing braces/brackets
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Failed to parse - return error with useful context
        safe_response = response[:200] if len(response) > 200 else response
        return {
            "tool": None,
            "error": f"Could not parse LLM response as JSON: {safe_response}"
        }
    
    def _sanitize_params(
        self,
        params: Dict[str, Any],
        tools_prompt: str,
        tool_name: str
    ) -> Dict[str, Any]:
        """
        Sanitize and fix common LLM parameter mistakes.
        
        UNIVERSAL DESIGN: Works with any database tool by fixing
        common type mismatches that LLMs make:
        - Arrays with single element when string expected
        - Empty strings that should be omitted
        - Type conversions (string numbers to actual numbers)
        - null/None values that should be omitted
        
        Args:
            params: Raw parameters from LLM
            tools_prompt: Tools prompt for schema lookup
            tool_name: Name of selected tool
            
        Returns:
            Sanitized parameters
        """
        import re
        
        # Known string parameters that LLMs sometimes wrap in arrays
        # These are always strings, never arrays
        # NOTE: "method" is NOT here because for 'explain' tool it IS an array of objects
        STRING_PARAMS = {"database", "collection", "newName", "projectId"}
        
        sanitized = {}
        
        for key, value in params.items():
            # Fix 0: Skip null/None values for optional parameters
            # LLMs often include "limit": null which databases reject
            if value is None:
                param_pattern = rf'{key}\*:'  # Required params have asterisk
                if not re.search(param_pattern, tools_prompt):
                    logger.debug(f"Sanitized {key}: skipping null optional value")
                    continue
            
            # Fix 1: Unwrap single-element arrays that should be scalar values
            # LLMs sometimes return {"database": ["test"]} instead of {"database": "test"}
            if isinstance(value, list) and len(value) == 1:
                # Check if this is a known string param OR not documented as array
                if key in STRING_PARAMS:
                    value = value[0]
                    logger.debug(f"Sanitized {key}: unwrapped known string param from array")
                else:
                    param_pattern = rf'{key}\*?:\s*array'
                    if not re.search(param_pattern, tools_prompt, re.IGNORECASE):
                        # Not documented as array, unwrap single element
                        value = value[0]
                        logger.debug(f"Sanitized {key}: unwrapped single-element array to scalar")
            
            # Fix 2: Convert numeric strings to actual numbers where appropriate
            if isinstance(value, str) and value.isdigit():
                param_pattern = rf'{key}\*?:\s*(integer|number)'
                if re.search(param_pattern, tools_prompt, re.IGNORECASE):
                    value = int(value)
                    logger.debug(f"Sanitized {key}: converted string to int")
            
            # Fix 3: Skip empty optional values that might cause issues
            # Keep empty strings/dicts for required params
            if value == "" or value == {} or value == []:
                param_pattern = rf'{key}\*:'  # Required params have asterisk
                if not re.search(param_pattern, tools_prompt):
                    logger.debug(f"Sanitized {key}: skipping empty optional value")
                    continue
            
            sanitized[key] = value
        
        return sanitized


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example of how to use QueryAgent"""
    
    from core.mcp.mongodb import MongoDBMCPClient
    
    # 1. Create and connect MongoDB client
    client = MongoDBMCPClient()
    await client.connect()
    
    # 2. Get tools prompt (same format as Zapier in OptimizedAgent!)
    tools_prompt = client.get_tools_prompt()
    
    # 3. Create Query Agent
    agent = QueryAgent()
    
    # 4. Execute natural language queries
    # This should ask for clarification since database is not specified
    result = await agent.execute(
        tools_prompt=tools_prompt,
        instruction="Add apple and mango to fruits collection",
        mcp_client=client
    )
    
    if result.needs_clarification:
        print(f"❓ Clarification needed: {result.clarification_message}")
        print(f"   Missing fields: {result.missing_fields}")
    elif result.success:
        print(f"✅ Success!")
        print(f"   Tool: {result.tool_name}")
        print(f"   Result: {result.result}")
    else:
        print(f"❌ Error: {result.error}")
    
    # 5. Execute with full info
    result = await agent.execute(
        tools_prompt=tools_prompt,
        instruction="Add apple to fruits collection in sample_database",
        mcp_client=client
    )
    
    print(f"Success: {result.success}")
    print(f"Tool: {result.tool_name}")
    print(f"Result: {result.result}")
    
    # Cleanup
    await client.disconnect()


if __name__ == "__main__":
    import asyncio
    
    # Load .env
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(example_usage())
