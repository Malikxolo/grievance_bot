"""
Customer Support Tool System
Boilerplate structure for support operations
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.last_used = None
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": self.name,
            "description": self.description,
            "usage_count": self.usage_count,
            "last_used": self.last_used
        }
    
    def _record_usage(self):
        """Record tool usage"""
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()


class LiveInformationTool(BaseTool):
    """Search for current real-time information (order status, product availability, etc.)"""
    
    def __init__(self, api_base_url: str = None, api_key: str = None):
        super().__init__(
            "live_information",
            "Get real-time order and customer information. Use for: order status lookup, tracking information, order history, customer profile. When customer asks about their order location, status, or delivery details."
        )
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.session = None
        
        logger.info("LiveInformationTool initialized")
    
    async def execute(self, query: str, user_id: str = None, 
                     query_type: str = "general", **kwargs) -> Dict[str, Any]:
        """
        Execute live information search
        
        Args:
            query: Search query (e.g., "order status for #12345", "is product XYZ available?")
            user_id: User identifier for personalized queries
            query_type: Type of query - "order_status", "product_availability", "inventory", "general"
        """
        self._record_usage()
        logger.info(f"Live information query: type={query_type}, query='{query[:50]}...'")
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Route to appropriate handler based on query type
            if query_type == "order_status":
                return await self._get_order_status(query, user_id)
            elif query_type == "product_availability":
                return await self._check_product_availability(query, user_id)
            elif query_type == "inventory":
                return await self._check_inventory(query, user_id)
            else:
                return await self._general_search(query, user_id)
                
        except Exception as e:
            logger.error(f"Live information query failed: {str(e)}")
            return {
                "success": False,
                "error": f"Live information query failed: {str(e)}",
                "query": query
            }
    
    async def _get_order_status(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Get order status from system"""
        
        # TODO: Implement order status API call
        # Example logic:
        # - Extract order ID from query
        # - Call order management API
        # - Return formatted order status
        
        logger.info(f"Fetching order status for query: {query}")
        
        # Placeholder implementation
        return {
            "success": True,
            "query": query,
            "query_type": "order_status",
            "data": {
                "order_id": "PLACEHOLDER",
                "status": "PENDING_IMPLEMENTATION",
                "message": "Order status lookup not yet implemented"
            }
        }
    
    async def _check_product_availability(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Check product availability"""
        
        # TODO: Implement product availability API call
        # Example logic:
        # - Extract product ID/name from query
        # - Call inventory API
        # - Return availability status
        
        logger.info(f"Checking product availability for query: {query}")
        
        # Placeholder implementation
        return {
            "success": True,
            "query": query,
            "query_type": "product_availability",
            "data": {
                "product": "PLACEHOLDER",
                "available": None,
                "message": "Product availability check not yet implemented"
            }
        }
    
    async def _check_inventory(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """Check inventory levels"""
        
        # TODO: Implement inventory API call
        # Example logic:
        # - Extract product/SKU from query
        # - Call warehouse/inventory API
        # - Return stock levels
        
        logger.info(f"Checking inventory for query: {query}")
        
        # Placeholder implementation
        return {
            "success": True,
            "query": query,
            "query_type": "inventory",
            "data": {
                "sku": "PLACEHOLDER",
                "stock_level": None,
                "message": "Inventory check not yet implemented"
            }
        }
    
    async def _general_search(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """General live information search"""
        
        # TODO: Implement general search logic
        # Example logic:
        # - Classify query intent
        # - Route to appropriate internal API
        # - Return relevant information
        
        logger.info(f"General live information search for query: {query}")
        
        # Placeholder implementation
        return {
            "success": True,
            "query": query,
            "query_type": "general",
            "data": {
                "message": "General live search not yet implemented"
            }
        }
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.debug("LiveInformationTool session closed")


class KnowledgeBaseTool(BaseTool):
    """Search internal documentation and FAQs"""
    
    def __init__(self, kb_config: Dict[str, Any] = None):
        super().__init__(
            "knowledge_base",
            "Search company documentation, FAQs, policies, and guides. Use for: return/refund policies, product information, how-to guides, shipping information. When customer asks policy questions or needs product help."
        )
        self.kb_config = kb_config or {}
        
        logger.info("KnowledgeBaseTool initialized")
    
    async def execute(self, query: str, user_id: str = None, 
                     category: str = None, **kwargs) -> Dict[str, Any]:
        """
        Execute knowledge base search
        
        Args:
            query: Search query
            user_id: User identifier for personalized results
            category: Optional category filter (e.g., "billing", "technical", "shipping")
        """
        self._record_usage()
        logger.info(f"Knowledge base query: category={category}, query='{query[:50]}...'")
        
        try:
            # TODO: Implement knowledge base search
            # Example logic:
            # - Use vector database (ChromaDB, Pinecone, etc.)
            # - Search internal documentation
            # - Rank results by relevance
            # - Return top matches with sources
            
            # Placeholder for vector search implementation
            results = await self._search_knowledge_base(query, category, user_id)
            
            return {
                "success": True,
                "query": query,
                "category": category,
                "results": results,
                "total_results": len(results)
            }
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {str(e)}")
            return {
                "success": False,
                "error": f"Knowledge base search failed: {str(e)}",
                "query": query
            }
    
    async def _search_knowledge_base(self, query: str, category: str = None, 
                                    user_id: str = None) -> List[Dict[str, Any]]:
        """Search knowledge base and return relevant articles"""
        
        # TODO: Implement vector search logic
        # Example implementation steps:
        # 1. Connect to vector database
        # 2. Generate query embedding
        # 3. Search for similar documents
        # 4. Filter by category if provided
        # 5. Format and return results
        
        logger.info(f"Searching knowledge base: query='{query}', category={category}")
        
        # Placeholder implementation
        return [
            {
                "title": "Placeholder Article",
                "content": "Knowledge base search not yet implemented",
                "category": category or "general",
                "relevance_score": 0.0,
                "article_id": "KB_PLACEHOLDER"
            }
        ]


class RaiseTicketTool(BaseTool):
    """Raise a support ticket in the system"""
    
    def __init__(self, ticketing_api_url: str = None, api_key: str = None):
        super().__init__(
            "raise_ticket",
            "Create support ticket for non-urgent issues requiring investigation. Use for: problems needing research, warehouse/courier checks, complex issues that can wait. When issue cannot be resolved immediately but is not urgent."
        )
        self.ticketing_api_url = ticketing_api_url
        self.api_key = api_key
        self.session = None
        
        logger.info("RaiseTicketTool initialized")
    
    async def execute(self, user_id: str, subject: str, description: str,
                     priority: str = "medium", category: str = "general",
                     metadata: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Raise a support ticket
        
        Args:
            user_id: User identifier
            subject: Ticket subject/title
            description: Detailed description of the issue
            priority: Ticket priority - "low", "medium", "high", "urgent"
            category: Ticket category - "technical", "billing", "general", "product"
            metadata: Additional metadata (order_id, product_id, etc.)
        """
        self._record_usage()
        logger.info(f"Raising ticket: user={user_id}, priority={priority}, category={category}")
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # TODO: Implement ticket creation logic
            # Example implementation:
            # - Validate input fields
            # - Call ticketing system API (Zendesk, Freshdesk, custom)
            # - Generate ticket ID
            # - Send confirmation
            # - Return ticket details
            
            ticket = await self._create_ticket(
                user_id, subject, description, priority, category, metadata
            )
            
            return {
                "success": True,
                "ticket_id": ticket.get("ticket_id"),
                "ticket": ticket,
                "message": "Ticket created successfully"
            }
            
        except Exception as e:
            logger.error(f"Ticket creation failed: {str(e)}")
            return {
                "success": False,
                "error": f"Ticket creation failed: {str(e)}",
                "user_id": user_id
            }
    
    async def _create_ticket(self, user_id: str, subject: str, description: str,
                           priority: str, category: str, 
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create ticket in ticketing system"""
        
        # TODO: Implement actual ticket creation
        # Example logic:
        # 1. Format ticket data according to ticketing system API
        # 2. Make POST request to create ticket
        # 3. Handle response and extract ticket ID
        # 4. Optionally send notification to user
        # 5. Return ticket details
        
        logger.info(f"Creating ticket for user {user_id}: {subject}")
        
        # Placeholder implementation
        ticket_data = {
            "ticket_id": "TICKET_PLACEHOLDER_12345",
            "user_id": user_id,
            "subject": subject,
            "description": description,
            "priority": priority,
            "category": category,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "message": "Ticket creation not yet fully implemented"
        }
        
        return ticket_data
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.debug("RaiseTicketTool session closed")


class AssignAgentTool(BaseTool):
    """Assign a human agent for follow-up"""
    
    def __init__(self, agent_api_url: str = None, api_key: str = None):
        super().__init__(
            "assign_agent",
            "Escalate to human agent for immediate assistance. Use for: urgent/critical issues, very frustrated customers, complex problems you cannot resolve, or when customer explicitly requests human agent. Provides immediate human support."
        )
        self.agent_api_url = agent_api_url
        self.api_key = api_key
        self.session = None
        
        logger.info("AssignAgentTool initialized")
    
    async def execute(self, user_id: str = None, issue_type: str = None, 
                     context: str = None, priority: str = "high",
                     preferred_channel: str = "chat", query: str = None, **kwargs) -> Dict[str, Any]:
        """
        Assign a human agent
        
        Args:
            user_id: User identifier
            issue_type: Type of issue - "technical", "billing", "complaint", "sales" (auto-detected if not provided)
            context: Context/history of the conversation
            priority: Assignment priority - "low", "medium", "high", "urgent"
            preferred_channel: Preferred communication channel - "chat", "phone", "email"
            query: Customer query to auto-detect issue type
        """
        self._record_usage()
        
        # Auto-detect issue type from query if not provided
        if not issue_type and query:
            query_lower = query.lower()
            if any(word in query_lower for word in ['refund', 'money', 'payment', 'charge', 'bill']):
                issue_type = "billing"
            elif any(word in query_lower for word in ['broken', 'defect', 'not working', 'issue', 'problem']):
                issue_type = "technical"
            elif any(word in query_lower for word in ['angry', 'frustrated', 'complaint', 'terrible', 'awful']):
                issue_type = "complaint"
            else:
                issue_type = "general"
        elif not issue_type:
            issue_type = "general"
        
        logger.info(f"Assigning agent: user={user_id}, type={issue_type}, priority={priority}")
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # TODO: Implement agent assignment logic
            # Example implementation:
            # - Check available agents based on issue type
            # - Consider agent workload and expertise
            # - Route to appropriate agent
            # - Create handoff record
            # - Notify agent and user
            # - Return assignment details
            
            assignment = await self._assign_to_agent(
                user_id, issue_type, context, priority, preferred_channel
            )
            
            return {
                "success": True,
                "assignment_id": assignment.get("assignment_id"),
                "agent_info": assignment.get("agent_info"),
                "estimated_response_time": assignment.get("eta"),
                "message": "Agent assigned successfully"
            }
            
        except Exception as e:
            logger.error(f"Agent assignment failed: {str(e)}")
            return {
                "success": False,
                "error": f"Agent assignment failed: {str(e)}",
                "user_id": user_id
            }
    
    async def _assign_to_agent(self, user_id: str, issue_type: str, 
                              context: str = None, priority: str = "medium",
                              preferred_channel: str = None) -> Dict[str, Any]:
        """Assign user to available agent"""
        
        # TODO: Implement agent routing logic
        # Example logic:
        # 1. Query available agents by expertise/issue_type
        # 2. Check agent availability and current workload
        # 3. Apply routing rules (round-robin, skills-based, priority)
        # 4. Create assignment in CRM/agent system
        # 5. Send notifications
        # 6. Return assignment details
        
        logger.info(f"Finding agent for user {user_id}: type={issue_type}, priority={priority}")
        
        # Placeholder implementation
        assignment_data = {
            "assignment_id": "ASSIGN_PLACEHOLDER_67890",
            "user_id": user_id,
            "issue_type": issue_type,
            "priority": priority,
            "agent_info": {
                "agent_id": "AGENT_PLACEHOLDER",
                "agent_name": "Placeholder Agent",
                "expertise": [issue_type],
                "status": "available"
            },
            "channel": preferred_channel or "chat",
            "eta": "5-10 minutes",
            "created_at": datetime.now().isoformat(),
            "context": context,
            "message": "Agent assignment not yet fully implemented"
        }
        
        return assignment_data
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.debug("AssignAgentTool session closed")


class OrderActionTool(BaseTool):
    """Execute order-related actions: refund, cancel, replace, return label"""
    
    def __init__(self, api_base_url: str = None, api_key: str = None):
        super().__init__(
            "order_action",
            "Execute order modifications: refund, cancel, replacement, return label, apply discount. IMPORTANT: For refunds/cancellations, use verification tool FIRST to check fraud risk. Available actions: refund, cancel, replace, return_label, apply_discount."
        )
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.session = None
        
        logger.info("OrderActionTool initialized")
    
    async def execute(self, action: str, order_id: str = None, user_id: str = None,
                     reason: str = None, amount: float = None, **kwargs) -> Dict[str, Any]:
        """
        Execute order actions
        
        Args:
            action: Action type - "refund", "cancel", "replace", "return_label", "apply_discount"
            order_id: Order identifier
            user_id: User identifier
            reason: Reason for action
            amount: Amount for refund/discount
        """
        self._record_usage()
        logger.info(f"Order action: action={action}, order_id={order_id}")
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Route to appropriate action handler
            if action == "refund":
                return await self._process_refund(order_id, user_id, amount, reason)
            elif action == "cancel":
                return await self._cancel_order(order_id, user_id, reason)
            elif action == "replace":
                return await self._create_replacement(order_id, user_id, reason)
            elif action == "return_label":
                return await self._generate_return_label(order_id, user_id, reason)
            elif action == "apply_discount":
                return await self._apply_discount(user_id, amount, reason)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "valid_actions": ["refund", "cancel", "replace", "return_label", "apply_discount"]
                }
                
        except Exception as e:
            logger.error(f"Order action failed: {str(e)}")
            return {
                "success": False,
                "error": f"Order action failed: {str(e)}",
                "action": action
            }
    
    async def _process_refund(self, order_id: str, user_id: str, amount: float, reason: str) -> Dict[str, Any]:
        """Process refund for order"""
        
        # TODO: Implement refund processing
        # Steps:
        # 1. Validate refund eligibility
        # 2. Calculate refund amount
        # 3. Call payment gateway API
        # 4. Update order status
        # 5. Send confirmation notification
        
        logger.info(f"Processing refund: order={order_id}, amount={amount}")
        
        return {
            "success": True,
            "action": "refund",
            "order_id": order_id,
            "refund_amount": amount,
            "status": "PENDING_IMPLEMENTATION",
            "message": "Refund processing not yet implemented"
        }
    
    async def _cancel_order(self, order_id: str, user_id: str, reason: str) -> Dict[str, Any]:
        """Cancel order"""
        
        # TODO: Implement order cancellation
        # Steps:
        # 1. Check if order can be cancelled (not shipped)
        # 2. Update order status to cancelled
        # 3. Initiate refund if already paid
        # 4. Update inventory
        # 5. Send confirmation
        
        logger.info(f"Cancelling order: order={order_id}")
        
        return {
            "success": True,
            "action": "cancel",
            "order_id": order_id,
            "status": "PENDING_IMPLEMENTATION",
            "message": "Order cancellation not yet implemented"
        }
    
    async def _create_replacement(self, order_id: str, user_id: str, reason: str) -> Dict[str, Any]:
        """Create replacement order"""
        
        # TODO: Implement replacement order creation
        # Steps:
        # 1. Verify original order
        # 2. Check inventory for replacement
        # 3. Create new order with same items
        # 4. Generate shipping label
        # 5. Link to original order
        # 6. Send confirmation
        
        logger.info(f"Creating replacement: order={order_id}")
        
        return {
            "success": True,
            "action": "replace",
            "original_order_id": order_id,
            "replacement_order_id": "PLACEHOLDER",
            "status": "PENDING_IMPLEMENTATION",
            "message": "Replacement order creation not yet implemented"
        }
    
    async def _generate_return_label(self, order_id: str, user_id: str, reason: str) -> Dict[str, Any]:
        """Generate return shipping label"""
        
        # TODO: Implement return label generation
        # Steps:
        # 1. Verify order is eligible for return
        # 2. Call shipping API to generate label
        # 3. Generate QR code for WhatsApp
        # 4. Store return authorization
        # 5. Send label to customer
        
        logger.info(f"Generating return label: order={order_id}")
        
        return {
            "success": True,
            "action": "return_label",
            "order_id": order_id,
            "label_url": "PLACEHOLDER",
            "tracking_number": "PLACEHOLDER",
            "status": "PENDING_IMPLEMENTATION",
            "message": "Return label generation not yet implemented"
        }
    
    async def _apply_discount(self, user_id: str, amount: float, reason: str) -> Dict[str, Any]:
        """Apply discount/credit to customer account"""
        
        # TODO: Implement discount application
        # Steps:
        # 1. Validate discount amount
        # 2. Generate coupon code or apply credit
        # 3. Update customer account
        # 4. Set expiration date
        # 5. Send notification
        
        logger.info(f"Applying discount: user={user_id}, amount={amount}")
        
        return {
            "success": True,
            "action": "apply_discount",
            "user_id": user_id,
            "discount_amount": amount,
            "coupon_code": "PLACEHOLDER",
            "status": "PENDING_IMPLEMENTATION",
            "message": "Discount application not yet implemented"
        }
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.debug("OrderActionTool session closed")


class VerificationTool(BaseTool):
    """Customer verification: Fraud detection and risk assessment (NO OTP - not needed for CS workflows)"""
    
    def __init__(self, fraud_api_key: str = None):
        super().__init__(
            "verification",
            "Check fraud risk and verify legitimacy of refund/cancel requests. ALWAYS use BEFORE order_action when processing refunds or cancellations. Returns risk_level (low/medium/high). If risk is high, use assign_agent instead of order_action."
        )
        self.fraud_api_key = fraud_api_key
        self.session = None
        
        logger.info("VerificationTool initialized")
    
    async def execute(self, query: str, user_id: str = None, 
                     order_id: str = None, action_requested: str = None, **kwargs) -> Dict[str, Any]:
        """
        Execute verification/fraud checks
        
        Args:
            query: Original customer query for context
            user_id: User identifier (phone number for WhatsApp)
            order_id: Order ID if action is order-related
            action_requested: Type of action customer wants (refund, cancel, etc.)
        """
        self._record_usage()
        logger.info(f"Verification check: user={user_id}, action={action_requested}")
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Perform fraud/risk assessment
            fraud_result = await self._fraud_check(user_id, order_id, action_requested, kwargs)
            
            return {
                "success": True,
                "verification_passed": fraud_result["risk_level"] != "high",
                "fraud_check": fraud_result,
                "message": "Verification check completed"
            }
                
        except Exception as e:
            logger.error(f"Verification action failed: {str(e)}")
            return {
                "success": False,
                "error": f"Verification action failed: {str(e)}",
                "verification_passed": False
            }
    
    async def _fraud_check(self, user_id: str, order_id: str, action: str, context: dict) -> Dict[str, Any]:
        """Check for fraud indicators"""
        
        # TODO: Implement fraud detection
        # Checks:
        # 1. Multiple refund requests in short time (rate limiting)
        # 2. High-value refund requests (flag for manual review)
        # 3. New account with immediate refund (suspicious pattern)
        # 4. Order history analysis (frequent returns = potential fraud)
        # 5. Blacklist checking (known fraudsters)
        # 6. Order value vs refund pattern
        
        logger.info(f"Fraud check: user={user_id}, order={order_id}, action={action}")
        
        # Placeholder implementation
        # In production, this would call fraud detection API or check Redis counters
        return {
            "fraud_score": 0.0,  # 0.0-1.0 scale
            "risk_level": "low",  # low, medium, high
            "indicators": [],  # List of fraud indicators found
            "recommendation": "proceed",  # proceed, manual_review, block
            "checks_performed": [
                "rate_limit_check",
                "order_history_check",
                "blacklist_check"
            ],
            "status": "PENDING_IMPLEMENTATION",
            "message": "Fraud detection not yet implemented - defaulting to low risk"
        }
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            logger.debug("VerificationTool session closed")


class ToolManager:
    """Manages all customer support tools"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tools: Dict[str, BaseTool] = {}
        self._initialize_tools()
        
        logger.info("ToolManager initialized")
    
    def _initialize_tools(self):
        """Initialize all configured tools"""
        
        # Live Information Tool (Customer & Order Data)
        live_info_config = self.config.get("live_information", {})
        self.tools["live_information"] = LiveInformationTool(
            api_base_url=live_info_config.get("api_url"),
            api_key=live_info_config.get("api_key")
        )
        
        # Knowledge Base Tool
        kb_config = self.config.get("knowledge_base", {})
        self.tools["knowledge_base"] = KnowledgeBaseTool(
            kb_config=kb_config
        )
        
        # Raise Ticket Tool
        ticket_config = self.config.get("raise_ticket", {})
        self.tools["raise_ticket"] = RaiseTicketTool(
            ticketing_api_url=ticket_config.get("api_url"),
            api_key=ticket_config.get("api_key")
        )
        
        # Assign Agent Tool
        agent_config = self.config.get("assign_agent", {})
        self.tools["assign_agent"] = AssignAgentTool(
            agent_api_url=agent_config.get("api_url"),
            api_key=agent_config.get("api_key")
        )
        
        # Order Action Tool (NEW)
        order_action_config = self.config.get("order_action", {})
        self.tools["order_action"] = OrderActionTool(
            api_base_url=order_action_config.get("api_url"),
            api_key=order_action_config.get("api_key")
        )
        
        # Verification Tool (NEW)
        verification_config = self.config.get("verification", {})
        self.tools["verification"] = VerificationTool(
            fraud_api_key=verification_config.get("fraud_api_key")
        )
        
        logger.info(f"✅ Initialized {len(self.tools)} tools: {list(self.tools.keys())}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools"""
        return {
            name: tool.description 
            for name, tool in self.tools.items()
        }
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name"""
        
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error(f"❌ Tool '{tool_name}' not available")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not available",
                "available_tools": self.get_available_tools()
            }
        
        try:
            logger.info(f"🔧 Executing tool: {tool_name}")
            result = await tool.execute(**kwargs)
            result["tool_name"] = tool_name
            
            if result.get("success"):
                logger.info(f"✅ Tool '{tool_name}' executed successfully")
            else:
                logger.warning(f"⚠️ Tool '{tool_name}' failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool '{tool_name}' execution error: {str(e)}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool_name": tool_name
            }
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools"""
        return {
            name: tool.get_info()
            for name, tool in self.tools.items()
        }
    
    async def cleanup(self):
        """Cleanup tool resources"""
        logger.info("🧹 Cleaning up tools...")
        
        for name, tool in self.tools.items():
            if hasattr(tool, 'close'):
                try:
                    await tool.close()
                    logger.debug(f"   ✅ Closed {name}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Error closing {name}: {str(e)}")
        
        logger.info("✅ Tool cleanup complete")