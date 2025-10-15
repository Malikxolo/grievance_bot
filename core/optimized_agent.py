"""
Optimized Single-Pass Agent System
Combines semantic analysis, tool execution, and response generation in minimal LLM calls
Now supports sequential tool execution with middleware for dependent tools
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

SYSTEM_PROMPT="""You are an expert analyst. Analyze queries using multi-signal intelligence covering semantics,
business opportunities, tool needs, and communication strategy. 

Your analysis must also demonstrate contextual intelligence — infer user intent, constraints, and emotional tone,
even if not explicitly stated.

Expand reasoning scope intelligently:
- Identify the user’s true problem or need beyond the literal query.
- Infer constraints such as budget, environment, urgency, or effort tolerance.
- Suggest adjacent or alternative solution paths when relevant. 
  (Example: If a user asks for "best AC on a tight budget", include air coolers or energy-efficient models.)
- When presenting solutions, prioritize durability, accessibility, and practicality.
- Briefly explain the reasoning or trade-off behind your choices, maintaining a natural and helpful tone.

Behavioral principles:
1. Curate, not list — provide reasoned, context-fitting answers rather than raw data.
2. Maintain an empathetic, peer-like tone — guide the user thoughtfully, not mechanically.
3. Adapt dynamically — balance between precision and user-centered flexibility.
4. Never invent irrelevant information, but always consider meaningful adjacent options.
5. Your reasoning framework:
   - What is the user's real goal or constraint?
   - What related domains or solutions could address it better?
   - How do I present this insight clearly, within structured output?

Output Format:
- Return valid JSON only.

Your mission: combine analytical precision with contextual empathy — deliver responses that are not only correct,
but *relevant, human, and insightful*.
"""



class OptimizedAgent:
    """Single-pass agent that minimizes LLM calls while maintaining all functionality"""
    
    def __init__(self, brain_llm, heart_llm, tool_manager):
        self.brain_llm = brain_llm
        self.heart_llm = heart_llm
        self.tool_manager = tool_manager
        self.available_tools = tool_manager.get_available_tools()
        logger.info(f"OptimizedAgent initialized with tools: {self.available_tools}")
    
    async def process_query(self, query: str, chat_history: List[Dict] = None, user_id: str = None) -> Dict[str, Any]:
        """Process query with minimal LLM calls"""
        logger.info(f" PROCESSING QUERY: '{query}'")
        start_time = datetime.now()
        
        logger.info(f" DEBUG CHAT HISTORY:")
        logger.info(f"   Type: {type(chat_history)}")
        logger.info(f"   Length: {len(chat_history) if chat_history else 0}")
        logger.info(f"   Content: {chat_history}")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Is None?: {chat_history is None}")
        
        try:
            # STEP 1: Single comprehensive analysis (combines semantic + business + planning + dependency detection)
            analysis_start = datetime.now()
            with open("debug_analysis_prompt.json", "w") as f:
                f.write(json.dumps(chat_history))
            analysis = await self._comprehensive_analysis(query, chat_history)
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            logger.info(f" Analysis completed in {analysis_time:.2f}s")
            
            # LOG: Enhanced analysis results
            logger.info(f" ANALYSIS RESULTS:")
            logger.info(f"   Intent: {analysis.get('semantic_intent', 'Unknown')}")
            business_opp = analysis.get('business_opportunity', {})
            logger.info(f"   Business Confidence: {business_opp.get('composite_confidence', 0)}/100")
            logger.info(f"   Engagement Level: {business_opp.get('engagement_level', 'none')}")
            logger.info(f"   Signal Breakdown: {business_opp.get('signal_breakdown', {})}")
            logger.info(f"   Tools Selected: {analysis.get('tools_to_use', [])}")
            logger.info(f"   Response Strategy: {analysis.get('response_strategy', {}).get('personality', 'Unknown')}")
            
            # LOG: Tool execution mode
            tool_execution = analysis.get('tool_execution', {})
            execution_mode = tool_execution.get('mode', 'parallel')
            logger.info(f"   Execution Mode: {execution_mode}")
            if execution_mode == 'sequential':
                logger.info(f"   Execution Order: {tool_execution.get('order', [])}")
                logger.info(f"   Dependency Reason: {tool_execution.get('dependency_reason', 'N/A')}")
            
            # STEP 2: Execute tools if needed (may include middleware LLM call for sequential)
            tool_start = datetime.now()
            tool_results = await self._execute_tools(
                analysis.get('tools_to_use', []),
                query,
                analysis,
                user_id
            )
            tool_time = (datetime.now() - tool_start).total_seconds()
            logger.info(f" Tools executed in {tool_time:.2f}s")
            
            if tool_results:
                logger.info(f" TOOL RESULTS SUMMARY:")
                for tool_name, result in tool_results.items():
                    if isinstance(result, dict) and result.get('success'):
                        logger.info(f"   {tool_name}: SUCCESS - {len(str(result))} chars of data")
                    elif isinstance(result, dict) and 'error' in result:
                        logger.info(f"   {tool_name}: ERROR - {result.get('error', 'Unknown')}")
                    else:
                        logger.info(f"   {tool_name}: RESULT - {type(result)} returned")
            else:
                logger.info(f" NO TOOLS EXECUTED - Conversational response only")
            
            response_start = datetime.now()
            logger.info(f" PASSING TO RESPONSE GENERATOR:")
            logger.info(f"   Analysis data: {len(str(analysis))} chars")
            logger.info(f"   Tool data: {len(str(tool_results))} chars")
            logger.info(f"   Strategy: {analysis.get('response_strategy', {})}")
            
            final_response = await self._generate_response(
                query,
                analysis,
                tool_results,
                chat_history
            )
            response_time = (datetime.now() - response_start).total_seconds()
            logger.info(f" Response generated in {response_time:.2f}s")
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Count actual LLM calls
            llm_calls = 2  # Brain + Heart
            if execution_mode == 'sequential':
                llm_calls += 1  # Middleware
            
            logger.info(f" TOTAL PROCESSING TIME: {total_time:.2f}s ({llm_calls} LLM calls)")
            
            return {
                "success": True,
                "response": final_response,
                "analysis": analysis,
                "tool_results": tool_results,
                "tools_used": analysis.get('tools_to_use', []),
                "execution_mode": execution_mode,
                "business_opportunity": analysis.get('business_opportunity', {}),
                "processing_time": {
                    "analysis": analysis_time,
                    "tools": tool_time,
                    "response": response_time,
                    "total": total_time
                }
            }
            
        except Exception as e:
            logger.error(f" Processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error. Please try again."
            }
    
    def _build_sentiment_language_guide(self, sentiment: Dict) -> str:
        """Build sentiment-driven language guidance"""
        emotion = sentiment.get('primary_emotion', 'casual')
        intensity = sentiment.get('intensity', 'medium')
        
        guides = {
            'frustrated': {
                'high': "User is highly frustrated - use very empathetic, understanding language. Be supportive and understanding", 
                'medium': "User is frustrated - be supportive and understanding",
                'low': "User is mildly frustrated - be gentle and reassuring"
            },
            'excited': {
                'high': "User is very excited - match their energy! Be enthusiastic and positive",
                'medium': "User is excited - be upbeat and encouraging",
                'low': "User is mildly excited - be positive and supportive"
            },
            'confused': {
                'high': "User is very confused - be extra patient and clear. Use simple language",
                'medium': "User is confused - be helpful and explanatory",
                'low': "User is slightly confused - be clarifying but not condescending"
            },
            'urgent': {
                'high': "User needs immediate help - be direct but supportive. Focus on solutions",
                'medium': "User has some urgency - be helpful and action-focused",
                'low': "User has mild urgency - be responsive and solution-oriented"
            },
            'casual': {
                'high': "User is very relaxed - be friendly and conversational",
                'medium': "User is casual - be warm and natural",
                'low': "User is somewhat casual - be friendly but focused"
            }
        }
        
        return guides.get(emotion, {}).get(intensity, "Be naturally helpful and friendly")

    async def _comprehensive_analysis(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Single LLM call for ALL analysis needs"""
        logger.info(f" ANALYSIS DEBUG:")
        logger.info(f"   Chat History Type: {type(chat_history)}")
        logger.info(f"   Chat History Content: {chat_history}")
        logger.info(f"   Chat History Length: {len(chat_history) if chat_history else 0}")
        
        # Build context from chat history
        context = self._build_context(chat_history)
        logger.info(f"   Built Context: '{context}'")
        
        # Create comprehensive prompt that does everything in one shot
        analysis_prompt = f"""Analyze this query using multi-signal intelligence and return a complete execution plan:

CONVERSATION CONTEXT: {context}
USER QUERY: {query}

AVAILABLE TOOLS:
- web_search: Search internet for current information
- rag: Retrieve from uploaded knowledge base  
- calculator: Perform calculations and analysis

Perform ALL of the following analyses in ONE response:

1. SEMANTIC INTENT (what user really wants)
   - What is the user trying to accomplish?
     * Solve a problem? (seeking solutions/recommendations)
     * Get information? (factual query, price check, comparison)
     * Learn something? (educational/conceptual)
     * Chat casually? (greetings, small talk)
   
   - If solving a problem:
     * What problem are they facing?
     * What solution did they mention, if any?
     * What constraints do they have?
   
   - Final intent: Based on the above, what should we help them with?
   
   Include every specific number, measurement, name, date, and technical detail from the user's query in your intent description. If they mentioned it, it's important - keep it

2. MOCHAND PRODUCT OPPORTUNITY ANALYSIS:

   WHAT MOCHAND DOES (context for understanding):
   - AI chatbot for customer communication automation
   - Supports WhatsApp, Facebook, Instagram messaging
   - Automates customer support, 24/7 availability
   - Handles repetitive customer queries
   - Integrates with CRM/payment systems
   
   DECISION LOGIC (strict matching):
   Step 1: Does query contain these EXACT keywords?
   - "customer" OR "client" OR "support" OR "service" OR "chat" OR "messaging" OR "communication"
   
   Step 2: Is user seeking solutions (not definitions)?
   
   If BOTH YES → business_opportunity.detected = true
   Otherwise → business_opportunity.detected = false
   
   EXCLUDE (even if problems mentioned):
   - Vague queries without keywords ("maintaining business", "business troubles")  
   - Other domains: accounting, inventory, HR, logistics

3. TOOL SELECTION:

    RAG SELECTION (STRICT RULE):
    Select rag if EITHER:
    1. Query is directly ABOUT Mochand (mentions "Mochand", "our/your product", "this chatbot", "what can you do", or asks about your capabilities/features)
    2. OR business_opportunity.detected = true

    Do NOT select rag for any other topic, even if you think RAG might contain related information
   
   GENERAL TOOL SELECTION:
   For web_search and calculator, analyze what information is needed:
   - What information is needed?
   - Where can it come from?
   - Select appropriate tools
   
   Use NO tools for:
   - Greetings, casual chat
   - General knowledge questions

4. SENTIMENT & PERSONALITY:
   - User's emotional state (frustrated/excited/casual/urgent/confused)
   - Best response personality (empathetic_friend/excited_buddy/helpful_dost/urgent_solver/patient_guide)

5. RESPONSE STRATEGY:
   - Response length (micro/short/medium/detailed)
   - Language style (hinglish/english/professional/casual)

6. DEPENDENCY & QUERY OPTIMIZATION:

    Step 1: If 2+ tools selected, detect dependencies

    DEFAULT = PARALLEL (tools run independently)

    Ask: "Does any tool need data from another tool's results to execute?"

    Common cases:
    - Calculator needs numbers/prices from web_search → SEQUENTIAL
    - Web_search needs info about "our product" from rag → SEQUENTIAL  
    - Two separate questions or topics → PARALLEL

    If any tool depends on another's output → mode="sequential"
    Otherwise → mode="parallel"

    Step 2: Generate queries based on POSITION in order array

    CRITICAL RULE FOR SEQUENTIAL MODE:
    → ONLY order[0] (first tool) gets a real query
    → ALL other tools (order[1], order[2], etc.) get "WAIT_FOR_PREVIOUS"
    → Do NOT generate real queries for any tool after the first one

    For tools that need real queries (first tool OR parallel mode):
    Query format rules:
    - RAG: "Mochand" + [topic from query]
    - calculator: Extract ONLY numbers from query, create valid Python math expression (no units, no text)
    - web_search: Transform the semantic_intent into a search query. Crucially, re-read the user's original query and preserve any words that specify WHEN (like "upcoming," "latest"), HOW MUCH (like "cheapest," "under 15000"), or WHAT TYPE (like "second-hand," "budget"). These qualifiers are critical to search accuracy. Add "2025" if time-sensitive.


    For tools that are waiting (order[1], order[2], etc. in sequential):
    - Set query to exactly: "WAIT_FOR_PREVIOUS"
    - Middleware will generate context-aware queries after previous tool completes

    Examples showing position-based logic:

    Example 1 - Sequential with 2 tools:
    Query: "compare our product with competitors"
    Analysis: web_search needs product info from rag → dependent
    Output:
    {{
    "mode": "sequential",
    "order": ["rag", "web_search"],
    "enhanced_queries": {{
        "rag": "Mochand features pricing",        ← Position 0: real query
        "web_search": "WAIT_FOR_PREVIOUS"         ← Position 1: wait
    }}
    }}

    Example 2 - Sequential with 3 tools:
    Query: "compare our product pricing to competitors and calculate difference"
    Analysis: All tools dependent in chain → sequential
    Output:
    {{
    "mode": "sequential",
    "order": ["rag", "web_search", "calculator"],
    "enhanced_queries": {{
        "rag": "Mochand pricing",                 ← Position 0: real query
        "web_search": "WAIT_FOR_PREVIOUS",        ← Position 1: wait
        "calculator": "WAIT_FOR_PREVIOUS"         ← Position 2: wait
    }}
    }}

    Example 3 - Parallel (completely independent):
    Query: "calculate 50 times 20"
    Analysis: Calculator has ALL numbers from query, no other tool needed → independent
    Output:
    {{
    "mode": "parallel",
    "enhanced_queries": {{
        "calculator": "50 * 20"
    }}
    }}

    Example 4 - Single tool (always parallel):
    Query: "what is my product?"
    Output:
    {{
    "mode": "parallel",
    "enhanced_queries": {{
        "rag": "Mochand product information"
    }}
    }}

Return ONLY valid JSON:
{{
    "semantic_intent": "clear description of what user wants",
    "business_opportunity": {{
        "detected": true/false,
        "composite_confidence": 0-100,
        "engagement_level": "direct_consultation|gentle_suggestion|empathetic_probing|pure_empathy",
        "signal_breakdown": {{
            "work_context": 0-100,
            "emotional_distress": 0-100,
            "solution_seeking": 0-100,
            "scale_scope": 0-100
        }},
        "pain_points": ["specific problems"],
        "solution_areas": ["how we can help"],
        "recommended_approach": "empathy_first|solution_focused|consultation_ready"
    }},
    "tools_to_use": ["tool1", "tool2"],
    "tool_execution": {{
        "mode": "sequential|parallel",
        "order": ["tool1", "tool2"],
        "dependency_reason": "why sequential is needed"
    }},
    "enhanced_queries": {{
        "web_search": "optimized search query or WAIT_FOR_PREVIOUS",
        "rag": "optimized rag query",
        "calculator": "clear calculation"
    }},
    "tool_reasoning": "why these tools",
    "sentiment": {{
        "primary_emotion": "frustrated|excited|casual|urgent|confused",
        "intensity": "low|medium|high"
    }},
    "response_strategy": {{
        "personality": "empathetic_friend|excited_buddy|helpful_dost|urgent_solver|patient_guide",
        "length": "micro|short|medium|detailed",
        "language": "hinglish|english|professional|casual",
        "tone": "friendly|professional|empathetic|excited"
    }},
    "key_points_to_address": ["point1", "point2"]
}}"""

        try:
            messages = []
            messages.append({"role": "user", "content": analysis_prompt})
            with open("debug_request_payload.json", "w") as f:
                json.dumps(messages)
            
            logger.info(f" CALLING BRAIN LLM for analysis...")
            response = await self.brain_llm.generate(
                messages,
                temperature=0.1,
                system_prompt="You are an expert analyst. Analyze queries using multi-signal intelligence covering semantics, business opportunities, tool needs, and communication strategy. Return valid JSON only." 
            )
            
            # LOG: Raw LLM response
            logger.info(f" BRAIN LLM RAW RESPONSE: {len(response)} chars")
            logger.info(f" First 200 chars: {response[:200]}...")
            
            # Clean response
            cleaned = self._clean_json_response(response)
            logger.info(f" CLEANED RESPONSE: {len(cleaned)} chars")
            
            analysis = json.loads(cleaned)
            
            # Ensure tool_execution exists with defaults
            if 'tool_execution' not in analysis:
                analysis['tool_execution'] = {
                    'mode': 'parallel',
                    'order': [],
                    'dependency_reason': ''
                }
            
            # LOG: Parsed analysis details
            logger.info(f" Analysis complete: intent={analysis.get('semantic_intent')}, "
                       f"business={analysis.get('business_opportunity', {}).get('detected')}, "
                       f"confidence={analysis.get('business_opportunity', {}).get('composite_confidence', 0)}, "
                       f"tools={analysis.get('tools_to_use', [])}")
            
            logger.info(f" FULL ANALYSIS GENERATED:")
            logger.info(f"   Semantic Intent: {analysis.get('semantic_intent', 'Unknown')}")
            logger.info(f"   Business Opportunity: {analysis.get('business_opportunity', {})}")
            logger.info(f"   Tools to Use: {analysis.get('tools_to_use', [])}")
            logger.info(f"   Tool Reasoning: {analysis.get('tool_reasoning', 'None')}")
            logger.info(f"   Sentiment: {analysis.get('sentiment', {})}")
            logger.info(f"   Response Strategy: {analysis.get('response_strategy', {})}")
            logger.info(f"   Enhanced Queries: {analysis.get('enhanced_queries', {})}")
            logger.info(f"   Tool Execution: {analysis.get('tool_execution', {})}")
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Return safe defaults
            return {
                "semantic_intent": "general_inquiry",
                "business_opportunity": {
                    "detected": False, 
                    "composite_confidence": 0,
                    "engagement_level": "pure_empathy",
                    "signal_breakdown": {
                        "work_context": 0,
                        "emotional_distress": 0,
                        "solution_seeking": 0,
                        "scale_scope": 0
                    },
                    "recommended_approach": "empathy_first"
                },
                "tools_to_use": [],
                "tool_execution": {
                    "mode": "parallel",
                    "order": [],
                    "dependency_reason": ""
                },
                "enhanced_queries": {},
                "sentiment": {"primary_emotion": "casual", "intensity": "medium"},
                "response_strategy": {
                    "personality": "helpful_dost",
                    "length": "medium",
                    "language": "hinglish",
                    "tone": "friendly"
                }
            }
    
    async def _execute_tools(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools with smart parallel/sequential handling based on dependencies"""
        
        if not tools:
            return {}
        
        # Check execution mode from analysis
        tool_execution = analysis.get('tool_execution', {})
        execution_mode = tool_execution.get('mode', 'parallel')
        
        # Route to appropriate execution method
        if execution_mode == 'sequential' and len(tools) > 1:
            logger.info(f" SEQUENTIAL EXECUTION MODE")
            return await self._execute_sequential(tools, query, analysis, user_id)
        else:
            logger.info(f" PARALLEL EXECUTION MODE")
            return await self._execute_parallel(tools, query, analysis, user_id)
    
    async def _execute_parallel(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools in parallel (original behavior)"""
        results = {}
        enhanced_queries = analysis.get('enhanced_queries', {})
        
        # Execute tools in parallel for speed
        tasks = []
        for tool in tools:
            if tool in self.available_tools:
                # Use enhanced query if available, fallback to raw query
                tool_query = enhanced_queries.get(tool, query)
                logger.info(f" {tool.upper()} ENHANCED QUERY: '{tool_query}'")
                
                task = self.tool_manager.execute_tool(tool, query=tool_query, user_id=user_id)
                tasks.append((tool, task))
        
        if tasks:
            # Gather all results in parallel
            for tool_name, task in tasks:
                try:
                    result = await task
                    results[tool_name] = result
                    logger.info(f" Tool {tool_name} executed successfully")
                except Exception as e:
                    logger.error(f" Tool {tool_name} failed: {e}")
                    results[tool_name] = {"error": str(e)}
        
        return results
    
    async def _execute_sequential(self, tools: List[str], query: str, analysis: Dict, user_id: str = None) -> Dict[str, Any]:
        """Execute tools sequentially with middleware for dependent queries"""
        results = {}
        enhanced_queries = analysis.get('enhanced_queries', {})
        tool_execution = analysis.get('tool_execution', {})
        order = tool_execution.get('order', tools)
        
        logger.info(f"   Execution order: {order}")
        logger.info(f"   Reason: {tool_execution.get('dependency_reason', 'N/A')}")
        
        # Execute first tool
        first_tool = order[0]
        first_query = enhanced_queries.get(first_tool, query)
        logger.info(f"   → Step 1: Executing {first_tool.upper()} with query: '{first_query}'")
        
        try:
            results[first_tool] = await self.tool_manager.execute_tool(first_tool, query=first_query, user_id=user_id)
            logger.info(f"   {first_tool} completed")
        except Exception as e:
            logger.error(f"   {first_tool} failed: {e}")
            results[first_tool] = {"error": str(e)}
            # Return early if first tool fails
            return results
        
        # Execute remaining tools with middleware
        for i in range(1, len(order)):
            current_tool = order[i]
            
            # Check if this tool needs middleware
            if enhanced_queries.get(current_tool) == "WAIT_FOR_PREVIOUS":
                logger.info(f"   → Step 2: Middleware generating enhanced query for {current_tool}...")
                
                # Use middleware to generate enhanced query from previous results
                enhanced_query = await self._middleware_summarizer(
                    previous_results=results,
                    original_query=query,
                    next_tool=current_tool
                )
                logger.info(f"   → Middleware output: '{enhanced_query}'")
            else:
                enhanced_query = enhanced_queries.get(current_tool, query)
            
            # Execute current tool
            logger.info(f"   → Step {i+2}: Executing {current_tool.upper()} with query: '{enhanced_query}'")
            try:
                results[current_tool] = await self.tool_manager.execute_tool(current_tool, query=enhanced_query, user_id=user_id)
                logger.info(f"  {current_tool} completed")
            except Exception as e:
                logger.error(f"   {current_tool} failed: {e}")
                results[current_tool] = {"error": str(e)}
        
        return results
    
    async def _middleware_summarizer(self, previous_results: Dict, original_query: str, next_tool: str) -> str:
        """Middleware: Extract key info from previous tool results and generate enhanced query"""
        
        # Format previous results
        previous_data = []
        for tool_name, result in previous_results.items():
            if isinstance(result, dict):
                if 'retrieved' in result:
                    previous_data.append(f"{tool_name.upper()} found: {result['retrieved'][:1000]}")
                elif 'results' in result and isinstance(result['results'], list):
                    for item in result['results'][:3]:
                        if 'snippet' in item:
                            previous_data.append(f"{tool_name.upper()}: {item['snippet'][:500]}")
        
        previous_summary = "\n".join(previous_data) if previous_data else "No data from previous tools"
        
        # SPECIAL HANDLING FOR CALCULATOR
        if next_tool == "calculator":
            middleware_prompt = f"""Extract numbers from data and create a math expression.

        ORIGINAL USER QUERY: {original_query}

        DATA FROM PREVIOUS TOOLS:
        {previous_summary}

        YOUR TASK:
        1. Find all numbers in the data above
        2. Understand what calculation the user wants from their query
        3. Create a valid Python math expression

        RULES:
        - Extract numbers only (remove ₹, $, %, commas)
        - Use operators: + - * / ( )
        - Match the calculation to user's query intent:
        * "total" or "sum" → add numbers
        * "difference" or "compare" → subtract
        * "multiply" or "times" → multiply
        * "percentage" or "discount" → multiply by decimal (15% = 0.15)
        * Complex queries → use parentheses for order

        EXAMPLES:
        Query: "compare prices", Data: "Item A: $2000, Item B: $1500"
        → "2000 - 1500"

        Query: "calculate 15% of 5000", Data: none needed
        → "5000 * 0.15"

        Query: "total cost for 3 items at 500 each", Data: "Price: ₹500"
        → "500 * 3"

        Query: "trip cost", Data: "Bus ₹600, Hotel ₹1000/night for 7 days, Food ₹200/day"
        → "600 + (1000*7) + (200*7)"

        Return ONLY a valid math expression. If you cannot determine what to calculate, return "SKIP"."""

        
        else:
           middleware_prompt = f"""Create a GENERIC industry search query based on previous data.

        ORIGINAL USER QUERY: {original_query}

        PREVIOUS TOOL RESULTS:
        {previous_summary}

        Your task: Create a GENERIC industry search query.
        Extract ONLY category/technology/use case (NOT specific company/product names).

        Examples:
        - Query: "compare competitors", Data: "CompanyX is a customer support chatbot for WhatsApp"
        → "customer support chatbot WhatsApp competitors 2025"

        - Query: "compare pricing", Data: "Product Y is an AI chatbot builder platform"
        → "AI chatbot builder pricing comparison 2025"

        - Query: "compare features", Data: "Our CRM system for sales teams"
        → "CRM sales teams features comparison 2025"

        Return ONLY the search query, nothing else. Keep it focused and under 10 words."""
        
        try:
            logger.info(f" Calling middleware LLM...")
            response = await self.brain_llm.generate(
                [{"role": "user", "content": middleware_prompt}],
                temperature=0.2,
                max_tokens=100
            )
            
            enhanced_query = response.strip()
            logger.info(f" Middleware generated: '{enhanced_query}'")
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Middleware failed: {e}")
            return original_query

    
    async def _generate_response(self, query: str, analysis: Dict, tool_results: Dict, chat_history: List[Dict]) -> str:
        """Generate response with simple business mode switching like old system"""
        
        # Extract key elements
        intent = analysis.get('semantic_intent', '')
        business_opp = analysis.get('business_opportunity', {})
        sentiment = analysis.get('sentiment', {})
        strategy = analysis.get('response_strategy', {})
        
        # Simple binary business mode logic (like your old system)
        business_detected = business_opp.get('detected', False)
        conversation_mode = "Smart Business Friend" if business_detected else "Casual Dost"
        
        # Actually USE the sentiment guide you built
        sentiment_guidance = self._build_sentiment_language_guide(sentiment)
        
        # Enhanced logging
        logger.info(f"  RESPONSE GENERATION INPUTS:")
        logger.info(f"   Intent: {intent}")
        logger.info(f"   Business Opportunity Detected: {business_detected}")
        logger.info(f"   Conversation Mode: {conversation_mode}")
        logger.info(f"   User Emotion: {sentiment.get('primary_emotion', 'casual')}")
        logger.info(f"   Sentiment Guidance: {sentiment_guidance}")
        logger.info(f"   Response Personality: {strategy.get('personality', 'helpful_dost')}")
        logger.info(f"   Response Length: {strategy.get('length', 'medium')}")
        logger.info(f"   Language Style: {strategy.get('language', 'hinglish')}")
        
        # Format tool results
        tool_data = self._format_tool_results(tool_results)
        logger.info(f" FORMATTED TOOL DATA: {len(tool_data)} chars")
        
        # Build memory context to avoid repetition
        recent_phrases = self._extract_recent_phrases(chat_history)
        logger.info(f" RECENT PHRASES TO AVOID: {recent_phrases}")
        
        # Simple, clean prompt (like your old system)
        response_prompt = f"""You are Mochand Dost - a naturally helpful AI friend who becomes a smart business consultant when needed.

    CURRENT CONVERSATION CONTEXT:
    - User Intent: {intent}
    - Business Opportunity Detected: {business_detected}
    - Conversation Mode: {conversation_mode}
    - User Sentiment Guide: {sentiment_guidance}
    - Pain Points: {business_opp.get('pain_points', [])}
    - Solution Areas: {business_opp.get('solution_areas', [])}

    AVAILABLE DATA TO USE NATURALLY:
    {tool_data}

    CONVERSATION MEMORY - Don't repeat these recent patterns:
    {recent_phrases}

    RESPONSE REQUIREMENTS:
    - Personality: {strategy.get('personality', 'helpful_dost')}
    - Length: {strategy.get('length', 'medium')} 
    - Language: {strategy.get('language', 'hinglish')}
    - Tone: {strategy.get('tone', 'friendly')}

    RESPONSE GENERATION RULES:
    1. NEVER restate or echo what the user just said
    2. NEVER announce tool usage ("Let me search...", "I found this data...")  
    3. Use the sentiment guide to match their emotional state perfectly
    4. If business opportunity detected, use progressive disclosure:
    - Early conversations: Focus on understanding their situation through questions
    - After they share pain points and seek guidance: Provide specific Mochand solutions
    - Match their engagement level naturally
    5. Stay in character as their helpful dost friend throughout
    6. End naturally based on context:
    - If user asks specific question → Provide info, optional question
    - If user seems to want guidance → Ask clarifying question
    - If providing detailed data → Just end with the info
    - Mix question and non-question endings for natural flow 

    USER QUERY: {query}

    Respond naturally as Mochand Dost in {conversation_mode} mode."""

        try:
            # Determine max tokens based on length strategy
            max_tokens = {
                "micro": 150,
                "short": 300,
                "medium": 500,
                "detailed": 700
            }.get(strategy.get('length', 'medium'), 500)
            
            logger.info(f" CALLING HEART LLM for response generation...")
            logger.info(f" Max tokens: {max_tokens}, Temperature: 0.4")
            
            messages = chat_history if chat_history else []
            messages.append({"role": "user", "content": response_prompt})
            with open("debug_response_prompt.json", "w") as f:
                f.write(json.dumps(messages))
            response = await self.heart_llm.generate(
                messages,
                temperature=0.4,
                max_tokens=4000,
                system_prompt="You are Mochand Dost, a conversational AI assistant. Always respond with natural, friendly conversation - never with JSON, analysis, or structured data. Be warm and helpful."
            )
            
            # LOG: Raw response from Heart LLM
            logger.info(f" HEART LLM RAW RESPONSE: {len(response)} chars")
            logger.info(f" First 200 chars: {response[:200]}...")
            
            # Clean and format
            response = self._clean_response(response)
            logger.info(f" FINAL CLEANED RESPONSE: {len(response)} chars")
            logger.info(f" FINAL RESPONSE: {response}")
            
            logger.info(f" Response generated: {len(response)} chars")
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I had trouble generating a response. Could you please try again?"
    
    def _build_context(self, chat_history: List[Dict]) -> str:
        """Build context from chat history"""
        if not chat_history:
            return "No previous context"
        
        recent = chat_history
        context_parts = []
        for item in recent:
            if 'query' in item:
                context_parts.append(f"User: {item['query']}")
            elif 'content' in item:
                context_parts.append(f"{item.get('role', 'user')}: {item['content']}")
        
        return " \n ".join(context_parts) if context_parts else "No previous context"
    
    def _format_tool_results(self, tool_results: dict) -> str:
        """Format tool results for response generation, handling different tool structures."""
        if not tool_results:
            return "No external data available"
        
        logger.info(f" RAW TOOL RESULTS DEBUG:")
        for tool_name, result in tool_results.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"TOOL: {tool_name.upper()}")
            logger.info(f"{'='*60}")
            
            if tool_name == 'web_search' and isinstance(result, dict):
                logger.info(f"Web Search Query: {result.get('query', 'N/A')}")
                logger.info(f"Success: {result.get('success', False)}")
                logger.info(f"Scraped Count: {result.get('scraped_count', 0)}") 
                
                if 'results' in result and isinstance(result['results'], list):
                    logger.info(f"Number of results: {len(result['results'])}")
                    
                    for idx, item in enumerate(result['results'][:5]):
                        logger.info(f"\n--- Result {idx+1} ---")
                        logger.info(f"Title: {item.get('title', 'No title')}")
                        logger.info(f"Snippet: {item.get('snippet', 'No snippet')}")
                        logger.info(f"Link: {item.get('link', 'No link')}")
                        
                        # scraped content
                        if 'scraped_content' in item:
                            scraped = item['scraped_content']
                            if scraped and not scraped.startswith("["):
                                logger.info(f"Scraped: {len(scraped)} chars")
                                logger.debug(f"Preview: {scraped[:200]}...")
                            else:
                                logger.info(f"Scraped: {scraped}")
        
        logger.info(f"\n{'='*60}\n")
        
        formatted = []
        
        for tool, result in tool_results.items():
            if isinstance(result, dict) and 'error' not in result:
                # Handle RAG-style result
                if 'success' in result and result['success']:
                    if 'retrieved' in result:
                        retrieved = result.get('retrieved', '')
                        chunks = result.get('chunks', [])
                        formatted.append(f"{tool.upper()} RETRIEVED TEXT:\n{retrieved}\n")
                        if chunks:
                            formatted.append(f"{tool.upper()} CHUNKS:\n" + "\n---\n".join(chunks))
                    
                    # Handle web search-style results
                    elif 'results' in result and isinstance(result['results'], list):
                        formatted.append(f"{tool.upper()} SEARCH RESULTS for query: {result.get('query', '')}\n")
                        for item in result['results']:
                            title = item.get('title', 'No title')
                            snippet = item.get('snippet', '')
                            link = item.get('link', '')
                            
                            # scraped content
                            if 'scraped_content' in item and item['scraped_content']:
                                scraped = item['scraped_content']
                                if not scraped.startswith("["):
                                    # UNIVERSAL CLEANUP - no char limit
                                    lines = scraped.split('\n')
                                    cleaned_lines = []
                                    
                                    for line in lines:
                                        line = line.strip()
                                        if len(line) < 40:  # Skip short lines (nav/menus)
                                            continue
                                        if line.count('http') > 2:  # Skip link lists
                                            continue
                                        if line.startswith('![') or line.startswith('Image'):  # Skip images
                                            continue
                                        cleaned_lines.append(line)
                                    
                                    # Join all cleaned lines
                                    cleaned = '\n'.join(cleaned_lines)
                                    
                                    formatted.append(f"- {title}\n  Content:\n{cleaned}\n  Link: {link}")
                                else:
                                    formatted.append(f"- {title}\n  {snippet}\n  Link: {link}")
                            else:
                                formatted.append(f"- {title}\n  {snippet}\n  Link: {link}")
                    
                    # Generic fallback for other data/result keys
                    elif 'data' in result:
                        formatted.append(f"{tool.upper()} DATA:\n{result['data']}")
                    elif 'result' in result:
                        formatted.append(f"{tool.upper()} RESULT:\n{result['result']}")
                    
                    else:
                        formatted.append(f"{tool.upper()}: Success but no recognizable content")
                else:
                    formatted.append(f"{tool.upper()}: No data retrieved or request failed")
            
            elif isinstance(result, str):
                formatted.append(f"{tool.upper()}: {result}")
        
        return "\n\n".join(formatted) if formatted else "No usable tool data"


    def _extract_recent_phrases(self, chat_history: List[Dict]) -> List[str]:
        """Extract recent phrases to avoid repetition"""
        if not chat_history:
            return []
        
        phrases = []
        for msg in chat_history[-4:]:
            if msg.get('role') == 'assistant' or 'response' in msg:
                content = msg.get('content', msg.get('response', ''))
                # Get first 30 chars of each sentence
                sentences = content.split('.')
                for sentence in sentences[:2]:
                    if sentence.strip():
                        phrases.append(sentence.strip()[:30])
        
        return phrases[-5:] if phrases else []
    
    def _clean_json_response(self, response: str) -> str:
        """Clean LLM response for JSON parsing"""
        response = response.strip()
        
        # Remove thinking tags
        if '<think>' in response:
            end_idx = response.find('</think>')
            if end_idx != -1:
                response = response[end_idx + 8:].strip()
        
        # Remove markdown blocks
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean final response for display"""
        # Remove any system thinking
        response = self._clean_json_response(response)
        
        # Fix formatting for display
        response = response.replace('- ', '-')
        response = response.strip()
        
        return response
