# Query Agent Implementation Plan

## 📋 Overview

A single-file Query Agent (`query_agent.py`) in `core/mcp/` that:

1. Receives tool list from MongoDBMCPClient (29 tools)
2. Receives natural language instruction from OptimizedAgent
3. Uses LLM (OpenRouter + Meta Llama) to select tool + generate params
4. Executes via MCP and returns result

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OptimizedAgent                               │
│  Detects DB intent → Calls nosql_query tool                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                │ nosql_query(
                                │   database="mongodb",
                                │   instruction="Add apple, mango to fruits"
                                │ )
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    QueryAgent (query_agent.py)                       │
│                                                                      │
│  Input:                                                              │
│    1. Tool List (from MongoDBMCPClient.list_tools())                │
│    2. Natural Language Instruction                                   │
│                                                                      │
│  Process:                                                            │
│    → Send to LLM (OpenRouter + Meta Llama)                          │
│    → LLM selects tool + generates params                            │
│                                                                      │
│  Output:                                                             │
│    → Execute selected tool via MCP                                   │
│    → Return result                                                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MongoDBMCPClient                                │
│                      execute_tool(name, params)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Location

```
core/mcp/
├── __init__.py
├── client.py
├── transport.py
├── mongodb.py
├── query_agent.py      ← NEW FILE (everything in one file)
├── security.py
├── exceptions.py
└── zapier_integration.py
```

---

## 🔧 QueryAgent Single File Structure

`core/mcp/query_agent.py` will contain:

1. **QueryAgent class** - Main agent
2. **System prompts** - Inside the same file
3. **LLM call logic** - OpenRouter API call
4. **Result parsing** - Parse LLM JSON response

---

## 🔑 Configuration

```
# From .env file
OPENROUTER_API_KEY=sk-or-...

# Model (hardcoded, change later)
MODEL = "meta-llama/llama-3.1-8b-instruct"  # or llama-3.1-70b
```

---

## 🔄 Execution Flow

### Input to QueryAgent:

```python
# 1. Tool list from MongoDB MCP (passed in)
tools = [
    {"name": "insert-one", "description": "Insert single document", "schema": {...}},
    {"name": "insert-many", "description": "Insert multiple documents", "schema": {...}},
    {"name": "find", "description": "Query documents", "schema": {...}},
    {"name": "aggregate", "description": "Run aggregation pipeline", "schema": {...}},
    # ... 29 tools total
]

# 2. Natural language instruction (from user via OptimizedAgent)
instruction = "Add apple and mango to fruits collection"
```

### LLM Prompt Construction:

```
System: You are a MongoDB query expert...

Available Tools:
- insert-one: Insert single document (database*, collection*, document*)
- insert-many: Insert multiple documents (database*, collection*, documents*)
- find: Query documents (database*, collection*, filter)
- aggregate: Run aggregation (database*, collection*, pipeline*)
... [all 29 tools]

User Instruction: "Add apple and mango to fruits collection"

Respond with JSON:
{
    "tool": "insert-many",
    "params": {
        "database": "default_db",
        "collection": "fruits",
        "documents": [{"name": "apple"}, {"name": "mango"}]
    }
}
```

### LLM Response:

```json
{
  "tool": "insert-many",
  "params": {
    "database": "default_db",
    "collection": "fruits",
    "documents": [{ "name": "apple" }, { "name": "mango" }]
  }
}
```

### Execute via MCP:

```python
result = await mongodb_client.execute_tool("insert-many", params)
```

---

## 📝 Implementation Phases

### Phase 1: Basic Structure

- [ ] Create `core/mcp/query_agent.py`
- [ ] QueryAgent class with `__init__` and `execute` methods
- [ ] Load OPENROUTER_API_KEY from .env
- [ ] Set model to Meta Llama

### Phase 2: Prompt Building

- [ ] Format tool list into prompt
- [ ] Build system prompt with all 29 tools
- [ ] Build user prompt with instruction

### Phase 3: LLM Integration

- [ ] Call OpenRouter API with prompt
- [ ] Parse JSON response
- [ ] Extract tool name and params

### Phase 4: Execution

- [ ] Call MongoDBMCPClient.execute_tool()
- [ ] Return result to OptimizedAgent

### Phase 5: Error Handling

- [ ] Handle invalid LLM response
- [ ] Handle MCP execution errors
- [ ] Retry logic if needed

### Phase 6: Integration

- [ ] Add `nosql_query` tool to OptimizedAgent
- [ ] End-to-end testing

---

## 🎯 Key Points

1. **Single File**: Everything in `query_agent.py` - prompts, LLM call, parsing
2. **Tool List**: Passed from MongoDBMCPClient (29 tools)
3. **Instruction**: Natural language from user
4. **LLM**: OpenRouter API + Meta Llama model
5. **Output**: Tool name + params → execute via MCP

---

## 🧪 Test Cases

| Instruction                  | Expected Tool | Expected Params                                   |
| ---------------------------- | ------------- | ------------------------------------------------- |
| "Add apple to fruits"        | insert-one    | {collection: "fruits", document: {name: "apple"}} |
| "Add apple, mango to fruits" | insert-many   | {collection: "fruits", documents: [...]}          |
| "Find all users"             | find          | {collection: "users", filter: {}}                 |
| "Find users older than 30"   | find          | {collection: "users", filter: {age: {$gt: 30}}}   |
| "Count products"             | count         | {collection: "products"}                          |
| "Delete expired sessions"    | delete-many   | {collection: "sessions", filter: {expired: true}} |

---

## ⚠️ Notes

- OpenRouter API key from `.env`
- Model: Meta Llama (configurable for future)
- Single LLM call for tool selection + param generation
- Tool list comes from `mongodb_client.list_tools()`
- No separate folders - everything in `core/mcp/query_agent.py`
