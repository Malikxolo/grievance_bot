from fastapi import FastAPI, APIRouter, UploadFile, File, Form, Depends, Body
from fastapi.responses import JSONResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis.asyncio import Redis
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage
from core.knowledge_base import (
    get_kb_manager,
    create_knowledge_base, 
    delete_collection,
    get_user_collections,
    add_files,
    delete_file
)
import logging
router = APIRouter()
import time
import os
from core import (
    LLMClient, HeartAgent, 
    ToolManager, Config, BrainAgent
)
from core.optimized_agent import OptimizedAgent
from core.logging_security import (
    safe_log_response,
    safe_log_user_data,
    safe_log_error,
    safe_log_query
)
import json
import shutil
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logging.info("⚡ Starting app lifespan...")
    
    global optimizedAgent
    config = Config()

    redis_client = Redis(
        host=os.getenv('REDIS_HOST'), 
        port=os.getenv('REDIS_PORT'), 
        decode_responses=os.getenv('REDIS_DECODE_RESPONSES'), 
        username=os.getenv('REDIS_USERNAME'), 
        password=os.getenv('REDIS_PASSWORD')
    )
    await FastAPILimiter.init(redis_client)
    
    brain_model_config = config.create_llm_config(
        provider=settings.brain_provider,
        model=settings.brain_model,
        max_tokens=1000
    )
    heart_model_config = config.create_llm_config(
        provider=settings.heart_provider,
        model=settings.heart_model,
        max_tokens=1000
    )
    web_model_config = config.get_tool_configs(
        web_model=settings.web_model,
        use_premium_search=settings.use_premium_search
    )

    brain_llm = LLMClient(brain_model_config)
    heart_llm = LLMClient(heart_model_config)
    tool_manager = ToolManager(config, brain_llm, web_model_config, settings.use_premium_search)

    optimizedAgent = OptimizedAgent(brain_llm, heart_llm, tool_manager)

    optimizedAgent.worker_task = asyncio.create_task(
        optimizedAgent.background_task_worker()
    )
    logging.info("OptimizedAgent background worker started")

    try:
        yield
    finally:
        logging.info("⚡ Shutting down app lifespan...")
        optimizedAgent.worker_task.cancel()
        try:
            await optimizedAgent.worker_task
        except asyncio.CancelledError:
            logging.info("OptimizedAgent worker cancelled cleanly")


from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryMessage(BaseModel):
    role: str
    content: str

class UserQuery(BaseModel):
    messages: List[QueryMessage]

class ChatMessage(BaseModel):
    userid: str
    chat_history: list[dict] = []
    user_query: str
    
from .global_config import settings

class UpdateAgentsRequest(BaseModel):
    brain_provider: Optional[str]
    brain_model: Optional[str]
    heart_provider: Optional[str]
    heart_model: Optional[str]
    use_premium_search: Optional[bool]
    web_model: Optional[str]


@router.post("/set_agents")
async def set_brain_heart_agents(request: UpdateAgentsRequest):
    """
    Update global Brain-Heart agents and web search configuration.
    """
    if request.brain_provider:
        settings.brain_provider = request.brain_provider
    if request.brain_model:
        settings.brain_model = request.brain_model
    if request.heart_provider:
        settings.heart_provider = request.heart_provider
    if request.heart_model:
        settings.heart_model = request.heart_model
    if request.use_premium_search is not None:
        settings.use_premium_search = request.use_premium_search
    if request.web_model:
        settings.web_model = request.web_model

    return JSONResponse(content={
        "status": "success",
        "current_settings": {
            "brain_provider": settings.brain_provider,
            "brain_model": settings.brain_model,
            "heart_provider": settings.heart_provider,
            "heart_model": settings.heart_model,
            "use_premium_search": settings.use_premium_search,
            "web_model": settings.web_model
        }
    })


@router.post("/chat", dependencies=[Depends(RateLimiter(times=6, seconds=60))])
async def chat_brain_heart_system(request: ChatMessage = Body(...)):
    """New endpoint specifically for Brain-Heart system with messages support"""
    
    try:
        user_id = request.userid
        user_query = request.user_query
        chat_history = request.chat_history if hasattr(request, 'chat_history') else []
        
        safe_log_user_data(user_id, 'brain_heart_chat', message_count=len(user_query))
        
        brain_provider = settings.brain_provider or os.getenv("BRAIN_LLM_PROVIDER")
        brain_model = settings.brain_model or os.getenv("BRAIN_LLM_MODEL")
        heart_provider = settings.heart_provider or os.getenv("HEART_LLM_PROVIDER")
        heart_model = settings.heart_model or os.getenv("HEART_LLM_MODEL")
        use_premium_search = settings.use_premium_search or os.getenv("USE_PREMIUM_SEARCH", "false").lower() == "true"
        web_model = settings.web_model or os.getenv("WEB_LLM_MODEL", "")
        
        config = Config()
        
        brain_model_config = config.create_llm_config(
            provider=brain_provider,
            model=brain_model,
            max_tokens=1000
        )
        heart_model_config = config.create_llm_config(
            provider=heart_provider,
            model=heart_model,
            max_tokens=1000
        )
        web_model_config = config.get_tool_configs(
            web_model=web_model,
            use_premium_search=use_premium_search
        )
        
        tool_manager = ToolManager(
            config,
            brain_model_config,
            web_model,
            use_premium_search
        )
        
        brain_llm = LLMClient(brain_model_config)
        heart_llm = LLMClient(heart_model_config)
        
        optimizedAgent = OptimizedAgent(
            brain_llm,
            heart_llm,
            tool_manager
        )
        
        result = await optimizedAgent.process_query(user_query, chat_history, user_id)
        
        if result["success"]:
            safe_log_response(result, level='info')
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(
                content={"error": result["error"]}, 
                status_code=500
            )
            
    except Exception as e:
        logging.error(f"❌ Brain-Heart chat endpoint failed: {str(e)}")
        return JSONResponse(
            content={"error": f"Brain-Heart processing failed: {str(e)}"}, 
            status_code=500
        )


# ============================================================================
# KNOWLEDGE BASE / COLLECTION ENDPOINTS (DATABASE-DRIVEN)
# ============================================================================

@router.get("/get-collections")
async def get_collections():
    """
    List all collections across all users from MongoDB.
    Returns aggregated view of all collections.
    """
    try:
        kb_mgr = get_kb_manager()
        
        if not kb_mgr.mongo_available:
            return JSONResponse(
                content={
                    "error": "MongoDB not available. Cannot retrieve collections.",
                    "collections": []
                }, 
                status_code=503
            )
        
        # Get all collections from MongoDB
        all_collections = list(kb_mgr.mongo_db.collections.find(
            {},
            {"_id": 0}
        ).sort("created_at", -1))
        
        # Format response
        formatted_collections = []
        for col in all_collections:
            metadata = col.copy()
            if "created_at" in metadata and hasattr(metadata["created_at"], "isoformat"):
                metadata["created_at"] = metadata["created_at"].isoformat()
            if "last_modified" in metadata and hasattr(metadata["last_modified"], "isoformat"):
                metadata["last_modified"] = metadata["last_modified"].isoformat()

            formatted_collections.append({
                "name": col.get("collection_name"),
                "user_id": col.get("user_id"),
                "metadata": metadata
            })
            
        return JSONResponse(
            content={"collections": formatted_collections}, 
            status_code=200
        )

        
    except Exception as e:
        logging.error(f"Error getting all collections: {e}")
        return JSONResponse(
            content={"error": str(e), "collections": []}, 
            status_code=500
        )


@router.get("/collections/{user_id}")
async def list_collections(user_id: str):
    """
    List all collections for a specific user from MongoDB.
    """
    try:
        collections = get_user_collections(user_id)
        
        formatted_data = []
        for col in collections:
            metadata = col.copy()
            if "created_at" in metadata and hasattr(metadata["created_at"], "isoformat"):
                metadata["created_at"] = metadata["created_at"].isoformat()
            if "last_modified" in metadata and hasattr(metadata["last_modified"], "isoformat"):
                metadata["last_modified"] = metadata["last_modified"].isoformat()

            formatted_data.append({
                "name": col.get("collection_name"),
                "metadata": metadata
            })

        
        return JSONResponse(
            content={"collections": formatted_data}, 
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error listing collections for user {user_id}: {e}")
        return JSONResponse(
            content={"error": str(e), "collections": []}, 
            status_code=500
        )


@router.get("/collections/{user_id}/{collection_name}")
async def get_collection_details(user_id: str, collection_name: str):
    """
    Get detailed information about a specific collection.
    """
    try:
        kb_mgr = get_kb_manager()
        collection_info = kb_mgr.get_collection_info(user_id, collection_name)
        
        if collection_info:
            metadata = collection_info.copy()
            if "created_at" in metadata and hasattr(metadata["created_at"], "isoformat"):
                metadata["created_at"] = metadata["created_at"].isoformat()
            if "last_modified" in metadata and hasattr(metadata["last_modified"], "isoformat"):
                metadata["last_modified"] = metadata["last_modified"].isoformat()

            return JSONResponse(
                content={"collection": metadata},
                status_code=200
            )

            
    except Exception as e:
        logging.error(f"Error getting collection details: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.post("/collections/create/local")
async def create_collection_local(
    user_id: str = Form(...),
    collection_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Create RAG collection from uploaded local files.
    Files are temporarily stored, processed, then deleted.
    """
    upload_dir = None
    try:
        # Create temporary upload directory
        upload_dir = f"temp_uploads/{user_id}/{collection_name}"
        os.makedirs(upload_dir, exist_ok=True)
        file_paths = []

        # Save uploaded files temporarily
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            file_paths.append(file_path)
        
        logging.info(f"Uploaded {len(file_paths)} files for user {user_id}")

        
        result = create_knowledge_base(user_id, collection_name, file_paths)

        if result.get("success"):
            metadata = result["metadata"].copy()
            metadata.pop("_id", None)
            if "created_at" in metadata:
                metadata["created_at"] = metadata["created_at"].isoformat()
            if "last_modified" in metadata:
                metadata["last_modified"] = metadata["last_modified"].isoformat()

            return JSONResponse(
                content={
                    "success": True,
                    "message": "Collection created successfully!",
                    "metadata": metadata
                },
                status_code=200
            )

        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=500
            )

    except Exception as e:
        logging.error(f"Error creating local collection: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )
    finally:
        # Cleanup temporary files
        if upload_dir and os.path.exists(upload_dir):
            shutil.rmtree(upload_dir, ignore_errors=True)
            logging.info(f"Cleaned up temporary directory: {upload_dir}")


@router.post("/collections/create/drive")
async def create_collection_drive(
    user_id: str = Form(...), 
    collection_name: str = Form(...), 
    files: List[dict] = Body(...)
):
    """
    Create RAG collection from Google Drive files.
    Supports multi-account downloads with conflict resolution.
    """
    downloaded_files = []
    try:
        from core.google_drive_integration import MultiAccountGoogleDriveManager

        drive_manager = MultiAccountGoogleDriveManager(user_id)

        # Download files from Google Drive
        downloaded_files = drive_manager.download_files_with_conflict_resolution(files)

        if not downloaded_files:
            return JSONResponse(
                content={"error": "Failed to download files from Google Drive"}, 
                status_code=500
            )

        logging.info(f"Downloaded {len(downloaded_files)} files from Google Drive")

        # Create knowledge base
        result = create_knowledge_base(user_id, collection_name, downloaded_files)

        # Revoke Drive sessions for security
        drive_manager.security_disconnect_all()

        if result.get("success"):
            metadata = result["metadata"].copy()
            metadata.pop("_id", None)
            if "created_at" in metadata:
                metadata["created_at"] = metadata["created_at"].isoformat()
            if "last_modified" in metadata:
                metadata["last_modified"] = metadata["last_modified"].isoformat()

            return JSONResponse(
                content={
                    "success": True,
                    "message": "Collection created successfully!",
                    "metadata": metadata
                },
                status_code=200
            )

        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=500
            )

    except Exception as e:
        logging.error(f"Error creating Drive collection: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )
    finally:
        # Cleanup downloaded files
        for fp in downloaded_files:
            if os.path.exists(fp):
                os.remove(fp)
                logging.info(f"Cleaned up downloaded file: {fp}")


@router.post("/collections/{user_id}/{collection_name}/add-files")
async def add_files_to_collection(
    user_id: str,
    collection_name: str,
    files: List[UploadFile] = File(...)
):
    """
    Add new files to an existing collection.
    Prevents duplicate file additions.
    """
    upload_dir = None
    try:
        # Create temporary upload directory
        upload_dir = f"temp_uploads/{user_id}/{collection_name}_add"
        os.makedirs(upload_dir, exist_ok=True)
        file_paths = []

        # Save uploaded files
        for file in files:
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            file_paths.append(file_path)

        # Add files to existing collection
        result = add_files(user_id, collection_name, file_paths)

        if result.get("success"):
            return JSONResponse(
                content={
                    "message": f"Added {result.get('added_chunks')} chunks from {len(result.get('added_files', []))} files",
                    "added_files": result.get("added_files"),
                    "total_chunks": result.get("total_chunks")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=500
            )

    except Exception as e:
        logging.error(f"Error adding files to collection: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )
    finally:
        # Cleanup
        if upload_dir and os.path.exists(upload_dir):
            shutil.rmtree(upload_dir, ignore_errors=True)


@router.delete("/collections/{user_id}/{collection_name}")
async def remove_collection(user_id: str, collection_name: str):
    """
    Delete an entire collection (removes from both ChromaDB and MongoDB).
    """
    try:
        result = delete_collection(user_id, collection_name)
        
        if result.get("success"):
            return JSONResponse(
                content={"message": f"Collection '{collection_name}' deleted successfully!"}, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=500
            )
            
    except Exception as e:
        logging.error(f"Error deleting collection: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.delete("/collections/{user_id}/{collection_name}/files/{filename}")
async def remove_file_from_collection(
    user_id: str, 
    collection_name: str, 
    filename: str
):
    """
    Delete a specific file from a collection.
    Removes all chunks associated with that file.
    """
    try:
        result = delete_file(user_id, collection_name, filename)
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "message": f"File '{filename}' deleted successfully!",
                    "deleted_chunks": result.get("deleted_chunks"),
                    "remaining_chunks": result.get("remaining_chunks")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=500
            )
            
    except Exception as e:
        logging.error(f"Error deleting file: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.post("/collections/{user_id}/{collection_name}/query")
async def query_collection(
    user_id: str,
    collection_name: str,
    query: str = Body(..., embed=True),
    n_results: int = Body(5, embed=True)
):
    """
    Query a specific collection for relevant information.
    Returns top N most relevant chunks.
    """
    try:
        from core.knowledge_base import query_knowledge_base
        
        result = query_knowledge_base(user_id, collection_name, query, n_results)
        
        if result.get("success"):
            return JSONResponse(
                content={
                    "query": result.get("query"),
                    "results": result.get("results"),
                    "metadatas": result.get("metadatas"),
                    "distances": result.get("distances")
                }, 
                status_code=200
            )
        else:
            return JSONResponse(
                content={"error": result.get("error")}, 
                status_code=500
            )
            
    except Exception as e:
        logging.error(f"Error querying collection: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@router.get("/health/knowledge-base")
async def health_check_kb():
    """
    Health check for knowledge base systems (ChromaDB + MongoDB).
    """
    try:
        kb_mgr = get_kb_manager()
        
        health_status = {
            "chromadb": "connected",
            "mongodb": "connected" if kb_mgr.mongo_available else "unavailable",
            "timestamp": time.time()
        }
        
        # Test ChromaDB
        try:
            kb_mgr.chroma_client.heartbeat()
        except Exception as e:
            health_status["chromadb"] = f"error: {str(e)}"
        
        # Test MongoDB
        if kb_mgr.mongo_available:
            try:
                kb_mgr.mongo_client.admin.command('ping')
            except Exception as e:
                health_status["mongodb"] = f"error: {str(e)}"
        
        status_code = 200 if health_status["chromadb"] == "connected" else 503
        
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )