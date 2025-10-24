"""
Knowledge Base Creation - Database-Driven Approach
Stores metadata in MongoDB, vectors in ChromaDB (no folder structure)
ENHANCED: PDF + DOCX Support + Multi-User Isolation
SECURITY: Input sanitization to prevent injection attacks
FIXED: PyMongo boolean check errors
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
import uuid
import tempfile
import shutil

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

# MongoDB for metadata storage
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from redis.asyncio import Redis

# Multi-format document loaders
try:
    from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
    PDF_SUPPORT = True
    DOCX_SUPPORT = True
except ImportError:
    PyPDFLoader = None
    UnstructuredWordDocumentLoader = None
    PDF_SUPPORT = False
    DOCX_SUPPORT = False
    
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

class UTF8TextLoader(TextLoader):
    """A custom TextLoader that enforces UTF-8 encoding."""
    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, encoding='utf-8', **kwargs)

class KnowledgeBaseManager:
    """Manages user-specific knowledge base creation with database storage"""
    
    def __init__(self, 
                 chroma_host: str = None,
                 chroma_port: int = 8000,
                 mongo_uri: str = None,
                 mongo_db_name: str = "knowledge_base"):
        """
        Initialize with database connections
        
        Args:
            chroma_host: ChromaDB host (if None, uses in-memory)
            chroma_port: ChromaDB port
            mongo_uri: MongoDB connection URI
            mongo_db_name: MongoDB database name
        """
        self.embeddings_model = "text-embedding-3-small"
        self.chunk_size = 1000
        self.chunk_overlap = 100
        
        # Setup ChromaDB client
        self.chroma_client = self._setup_chroma_client(chroma_host, chroma_port)
        
        # Setup MongoDB client for metadata
        self.mongo_client, self.mongo_db = self._setup_mongo_client(mongo_uri, mongo_db_name)
        
        self.mongo_available = self.mongo_db is not None
        
        self.redis_client = Redis(
            host=os.getenv('REDIS_HOST'), 
            port=os.getenv('REDIS_PORT'), 
            decode_responses=os.getenv('REDIS_DECODE_RESPONSES'), 
            username=os.getenv('REDIS_USERNAME'), 
            password=os.getenv('REDIS_PASSWORD')
        )
        
        logger.info("KnowledgeBaseManager initialized with database storage")
        if PDF_SUPPORT:
            logger.info("PDF support available")
        if DOCX_SUPPORT:
            logger.info("DOCX support available")
    
    def _setup_chroma_client(self, host: str = None, port: int = 8000):
        """Setup ChromaDB client"""
        try:
            if host or os.getenv('CHROMA_HOST'):
                # Use hosted ChromaDB
                actual_host = host or os.getenv('CHROMA_HOST')
                actual_port = port or int(os.getenv('CHROMA_PORT', 8000))
                logger.info(f"Using ChromaDB server: {actual_host}:{actual_port}")
                return chromadb.HttpClient(host=actual_host, port=actual_port)
            else:
                
                logger.info("Using local ChromaDB (disk persistence)")
                return chromadb.PersistentClient()
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB: {e}")
            raise
        
    def _set_active_collection(self, collection_name: str, user_id: str):
        """Set active ChromaDB collection for user"""
        namespaced_name = self._get_namespaced_collection_name(user_id, collection_name)
        return self.redis_client.set(f"user:{user_id}:active_collection", namespaced_name)
    
    def _get_active_collection(self, user_id: str) -> Optional[str]:
        """Get active ChromaDB collection for user"""
        return self.redis_client.get(f"user:{user_id}:active_collection")
    
    def _setup_mongo_client(self, uri: str = None, db_name: str = "knowledge_base"):
        """Setup MongoDB client for metadata storage"""
        try:
            mongo_uri = uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            client.admin.command('ping')
            db = client[db_name]
            
            # Create indexes for efficient queries
            db.collections.create_index([("user_id", 1), ("collection_name", 1)], unique=True)
            db.collections.create_index([("user_id", 1)])
            db.collections.create_index([("created_at", -1)])
            
            logger.info(f"Connected to MongoDB: {mongo_uri}")
            return client, db
            
        except ConnectionFailure as e:
            logger.warning(f"MongoDB connection failed: {e}. Metadata storage disabled.")
            return None, None
        except Exception as e:
            logger.warning(f"MongoDB setup failed: {e}. Metadata storage disabled.")
            return None, None
    
    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name to prevent injection"""
        # Remove dangerous characters, keep alphanumeric, underscore, hyphen
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        # Limit length
        return sanitized[:100]
    
    def _get_namespaced_collection_name(self, user_id: str, collection_name: str) -> str:
        """Create user-namespaced collection name"""
        safe_user_id = self._sanitize_collection_name(user_id)
        safe_collection = self._sanitize_collection_name(collection_name)
        return f"{safe_user_id}_{safe_collection}"
    
    def create_user_knowledge_base(
        self, 
        user_id: str, 
        collection_name: str, 
        file_paths: List[str]
    ) -> Dict[str, Any]:
        """Create user-specific knowledge base from uploaded files"""
        try:
            logger.info(f"Starting knowledge base creation for user {user_id}, collection: {collection_name}")
            
            # Sanitize inputs
            safe_user_id = self._sanitize_collection_name(user_id)
            safe_collection_name = self._sanitize_collection_name(collection_name)
            
            # Extract file names
            file_names = [os.path.basename(fp) for fp in file_paths]
            
            # Check if collection already exists
            if self.mongo_available:  # FIXED: Use boolean flag
                existing = self.mongo_db.collections.find_one({
                    "user_id": safe_user_id,
                    "collection_name": safe_collection_name
                })
                if existing is not None:  # FIXED: Compare with None
                    return {
                        "success": False,
                        "error": f"Collection '{safe_collection_name}' already exists"
                    }
            
            # Load documents from uploaded files
            logger.info(f"Loading {len(file_paths)} documents...")
            documents = self._load_documents_from_files(file_paths)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents could be loaded",
                    "details": {
                        "file_count": len(file_paths),
                        "file_names": file_names
                    }
                }
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks...")
            texts = self._split_documents(documents)
            logger.info(f"Split into {len(texts)} chunks")
            
            # Create embeddings and vector store
            logger.info("Creating embeddings and ChromaDB vector store...")
            result = self._create_vector_store(texts, safe_user_id, safe_collection_name)
            self._set_active_collection(safe_collection_name, safe_user_id)
            if result["success"]:
                # Store metadata in MongoDB
                metadata = {
                    "user_id": safe_user_id,
                    "collection_name": safe_collection_name,
                    "snippet": texts[0].page_content[:50] if texts else "",
                    "file_count": len(file_paths),
                    "file_names": file_names,
                    "document_count": len(documents),
                    "chunk_count": len(texts),
                    "embedding_model": self.embeddings_model,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "created_at": datetime.now(timezone.utc),
                    "last_modified": datetime.now(timezone.utc)
                }
                
                self._save_metadata(metadata)
                
                logger.info(f"Knowledge base created successfully: {safe_collection_name}")
                return {
                    "success": True,
                    "message": f"Knowledge base '{safe_collection_name}' created successfully",
                    "metadata": metadata
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Failed to create knowledge base: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Knowledge base creation failed: {str(e)}"
            }
            
    def delete_knowledge_base(self, user_id: str, collection_name: str) -> Dict[str, Any]:
        """Delete entire user-specific knowledge base"""
        try:
            safe_user_id = self._sanitize_collection_name(user_id)
            safe_collection_name = self._sanitize_collection_name(collection_name)
            
            # Delete from ChromaDB
            namespaced_name = self._get_namespaced_collection_name(safe_user_id, safe_collection_name)
            self._delete_vector_store(namespaced_name)
            
            # Delete from MongoDB
            if self.mongo_available:  # FIXED: Use boolean flag
                result = self.mongo_db.collections.delete_one({
                    "user_id": safe_user_id,
                    "collection_name": safe_collection_name
                })
                logger.info(f"Deleted metadata: {result.deleted_count} document(s)")
            
            return {
                "success": True,
                "message": f"Knowledge base '{safe_collection_name}' deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting knowledge base: {e}")
            return {
                "success": False,
                "error": f"Failed to delete knowledge base: {str(e)}"
            }
    
    def _load_documents_from_files(self, file_paths: List[str]) -> List[Document]:
        """Load documents from uploaded files"""
        documents = []
        
        # Try to import unstructured for comprehensive format support
        try:
            from unstructured.partition.auto import partition
            UNSTRUCTURED_AVAILABLE = True
            logger.info("Unstructured package available")
        except ImportError:
            UNSTRUCTURED_AVAILABLE = False
            logger.warning("Unstructured not available. Limited format support.")
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            filename = os.path.basename(file_path)
            file_extension = os.path.splitext(filename)[1].lower()
            
            logger.info(f"Processing file: {filename} (type: {file_extension})")
            
            try:
                # Use unstructured for comprehensive format support
                if UNSTRUCTURED_AVAILABLE and file_extension in [
                    '.md', '.rtf', '.csv', '.tsv', '.json',
                    '.xlsx', '.xls', '.xlsm', '.xlsb', '.xltx', '.xltm',
                    '.docm', '.dotx', '.dotm', '.dot',
                    '.ppt', '.pptx', '.pptm', '.potx', '.potm', '.ppsx', '.ppsm',
                    '.html', '.htm', '.xml',
                    '.odt', '.ods', '.odp',
                    '.epub', '.msg', '.eml'
                ]:
                    elements = partition(filename=file_path)
                    content = "\n\n".join([str(element) for element in elements])
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": filename,
                            "file_type": file_extension,
                            "elements_count": len(elements)
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Loaded {file_extension} file: {filename} ({len(elements)} elements)")
                    
                elif file_extension == '.txt':
                    loader = UTF8TextLoader(file_path)
                    file_docs = loader.load()
                    for doc in file_docs:
                        doc.metadata["source"] = filename
                    documents.extend(file_docs)
                    logger.info(f"Loaded text file: {filename}")
                    
                elif file_extension == '.pdf':
                    if PDF_SUPPORT:
                        loader = PyPDFLoader(file_path)
                        file_docs = loader.load()
                        for doc in file_docs:
                            doc.metadata["source"] = filename
                        documents.extend(file_docs)
                        logger.info(f"Loaded PDF: {filename} ({len(file_docs)} pages)")
                    else:
                        logger.error(f"PDF support not available")
                        
                elif file_extension in ['.doc', '.docx']:
                    if DOCX_SUPPORT:
                        loader = UnstructuredWordDocumentLoader(file_path)
                        file_docs = loader.load()
                        for doc in file_docs:
                            doc.metadata["source"] = filename
                        documents.extend(file_docs)
                        logger.info(f"Loaded Word document: {filename}")
                    else:
                        logger.error(f"DOCX support not available")
                        
                else:
                    logger.warning(f"Unsupported file type: {file_extension} for file {filename}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Detect encryption/password-protected files
                encryption_keywords = [
                    'encrypt', 'password', 'decrypt', 'protected', 
                    'badzipfile', 'pdfdecryptionerror', 'pdfreadeerror',
                    'bad magic number', 'file is encrypted', 'bad password'
                ]
                
                if any(keyword in error_msg for keyword in encryption_keywords):
                    logger.error(f"❌ ENCRYPTED FILE: '{filename}' is password-protected")
                else:
                    logger.error(f"Error processing file {filename}: {e}")
                
                continue
        
        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def _delete_vector_store(self, collection_name: str):
        """Delete ChromaDB collection"""
        try:
            existing_collections = [c.name for c in self.chroma_client.list_collections()]
            if collection_name in existing_collections:
                self.chroma_client.delete_collection(name=collection_name)
                logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting vector store: {e}")
    
    def _create_vector_store(self, texts: List[Document], user_id: str, collection_name: str) -> Dict[str, Any]:
        """Create ChromaDB vector store"""
        try:
            # Create embeddings
            embeddings = OpenAIEmbeddings(
                model=self.embeddings_model, 
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # Create namespaced collection name
            namespaced_name = self._get_namespaced_collection_name(user_id, collection_name)
            
            # Clean up existing collection
            self._delete_vector_store(namespaced_name)
            
            # Create new collection
            collection = self.chroma_client.get_or_create_collection(name=namespaced_name)
            
            # Generate embeddings for all texts
            contents = [doc.page_content for doc in texts]
            metadatas = [doc.metadata for doc in texts]
            
            # Create embeddings
            embeddings_list = embeddings.embed_documents(contents)
            
            # Generate IDs
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Add to collection
            collection.add(
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings_list,
                ids=ids
            )
            
            # Verify
            count = collection.count()
            logger.info(f"Vector store created: {count} entries in collection '{namespaced_name}'")
            
            return {
                "success": True,
                "collection_count": count,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Vector store creation failed: {str(e)}"
            }
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to MongoDB"""
        if not self.mongo_available:  # FIXED: Use boolean flag
            logger.warning("MongoDB not available, metadata not saved")
            return
        
        try:
            self.mongo_db.collections.insert_one(metadata)
            logger.info(f"Metadata saved for collection: {metadata['collection_name']}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_collection_info(self, user_id: str, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about user's collection"""
        if not self.mongo_available:  # FIXED: Use boolean flag
            return None
        
        try:
            safe_user_id = self._sanitize_collection_name(user_id)
            safe_collection_name = self._sanitize_collection_name(collection_name)
            
            metadata = self.mongo_db.collections.find_one({
                "user_id": safe_user_id,
                "collection_name": safe_collection_name
            }, {"_id": 0})
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None
    
    def get_user_collections(self, user_id: str) -> List[Dict[str, Any]]:
        """Get list of user's collections with metadata"""
        if not self.mongo_available:  # FIXED: Use boolean flag
            return []
        
        try:
            safe_user_id = self._sanitize_collection_name(user_id)
            
            collections = list(self.mongo_db.collections.find(
                {"user_id": safe_user_id},
                {"_id": 0}
            ).sort("created_at", -1))
            
            logger.info(f"Found {len(collections)} collections for user {safe_user_id}")
            return collections
            
        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []
    
    def query_collection(
        self, 
        user_id: str, 
        collection_name: str, 
        query: str, 
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Query user's knowledge base collection"""
        try:
            safe_user_id = self._sanitize_collection_name(user_id)
            safe_collection_name = self._sanitize_collection_name(collection_name)
            
            # Get namespaced collection
            namespaced_name = self._get_namespaced_collection_name(safe_user_id, safe_collection_name)
            collection = self.chroma_client.get_collection(name=namespaced_name)
            
            # Create query embedding
            embeddings = OpenAIEmbeddings(
                model=self.embeddings_model,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            query_embedding = embeddings.embed_query(query)
            
            # Perform query
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            logger.info(f"Query executed: {len(results['documents'][0])} results")
            
            return {
                "success": True,
                "query": query,
                "results": results['documents'][0] if results['documents'] else [],
                "metadatas": results['metadatas'][0] if results['metadatas'] else [],
                "distances": results['distances'][0] if results['distances'] else []
            }
            
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return {
                "success": False,
                "error": f"Query failed: {str(e)}",
                "results": []
            }

    def delete_file_from_collection(
        self, 
        user_id: str, 
        collection_name: str, 
        filename: str
    ) -> Dict[str, Any]:
        """Delete all chunks from a specific file"""
        try:
            safe_user_id = self._sanitize_collection_name(user_id)
            safe_collection_name = self._sanitize_collection_name(collection_name)
            
            namespaced_name = self._get_namespaced_collection_name(safe_user_id, safe_collection_name)
            collection = self.chroma_client.get_collection(name=namespaced_name)
            
            count_before = collection.count()
            
            # Get and delete items
            items_to_delete = collection.get(where={"source": filename})
            if items_to_delete['ids']:
                collection.delete(ids=items_to_delete['ids'])
            
            count_after = collection.count()
            deleted_chunks = count_before - count_after
            
            # Update MongoDB metadata
            if self.mongo_available:  # FIXED: Use boolean flag
                self.mongo_db.collections.update_one(
                    {"user_id": safe_user_id, "collection_name": safe_collection_name},
                    {
                        "$pull": {"file_names": filename},
                        "$inc": {
                            "file_count": -1,
                            "chunk_count": -deleted_chunks
                        },
                        "$set": {"last_modified": datetime.utcnow()}
                    }
                )
            
            logger.info(f"Deleted {deleted_chunks} chunks from file '{filename}'")
            
            return {
                "success": True,
                "deleted_file": filename,
                "deleted_chunks": deleted_chunks,
                "remaining_chunks": count_after
            }
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return {
                "success": False,
                "error": f"Failed to delete file: {str(e)}"
            }

    def add_files_to_collection(
        self, 
        user_id: str, 
        collection_name: str, 
        file_paths: List[str]
    ) -> Dict[str, Any]:
        """Add new files to existing collection"""
        try:
            safe_user_id = self._sanitize_collection_name(user_id)
            safe_collection_name = self._sanitize_collection_name(collection_name)
            
            # Check for duplicates
            if self.mongo_available:  # FIXED: Use boolean flag
                existing_metadata = self.mongo_db.collections.find_one({
                    "user_id": safe_user_id,
                    "collection_name": safe_collection_name
                })
                
                if existing_metadata is not None:  # FIXED: Compare with None
                    existing_files = set(existing_metadata.get("file_names", []))
                    new_file_names = [os.path.basename(fp) for fp in file_paths]
                    duplicates = [f for f in new_file_names if f in existing_files]
                    
                    if duplicates:
                        return {
                            "success": False,
                            "error": f"Files already exist: {', '.join(duplicates)}"
                        }
            
            # Load and process new files
            documents = self._load_documents_from_files(file_paths)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents could be loaded"
                }
            
            # Split into chunks
            texts = self._split_documents(documents)
            
            # Get existing collection
            namespaced_name = self._get_namespaced_collection_name(safe_user_id, safe_collection_name)
            collection = self.chroma_client.get_collection(name=namespaced_name)
            
            current_count = collection.count()
            
            # Create embeddings
            embeddings = OpenAIEmbeddings(
                model=self.embeddings_model,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            contents = [doc.page_content for doc in texts]
            metadatas = [doc.metadata for doc in texts]
            embeddings_list = embeddings.embed_documents(contents)
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Add to collection
            collection.add(
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings_list,
                ids=ids
            )
            
            # Update MongoDB metadata
            file_names = [os.path.basename(fp) for fp in file_paths]
            if self.mongo_available:  # FIXED: Use boolean flag
                self.mongo_db.collections.update_one(
                    {"user_id": safe_user_id, "collection_name": safe_collection_name},
                    {
                        "$push": {"file_names": {"$each": file_names}},
                        "$inc": {
                            "file_count": len(file_names),
                            "document_count": len(documents),
                            "chunk_count": len(texts)
                        },
                        "$set": {"last_modified": datetime.utcnow()}
                    }
                )
            
            return {
                "success": True,
                "added_files": file_names,
                "added_chunks": len(texts),
                "total_chunks": current_count + len(texts)
            }
            
        except Exception as e:
            logger.error(f"Error adding files: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to add files: {str(e)}"
            }


# Convenience functions
_kb_manager = None

def get_kb_manager():
    """Get or create singleton KnowledgeBaseManager"""
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
    return _kb_manager

def create_knowledge_base(user_id: str, collection_name: str, file_paths: List[str]) -> Dict[str, Any]:
    return get_kb_manager().create_user_knowledge_base(user_id, collection_name, file_paths)

def query_knowledge_base(user_id: str, collection_name: str, query: str, n_results: int = 5) -> Dict[str, Any]:
    return get_kb_manager().query_collection(user_id, collection_name, query, n_results)

def get_active_collection(user_id: str) -> Optional[str]:
    return get_kb_manager()._get_active_collection(user_id)

def set_active_collection(user_id: str, collection_name: str):
    return get_kb_manager()._set_active_collection(collection_name, user_id)

def get_user_collections(user_id: str) -> List[Dict[str, Any]]:
    return get_kb_manager().get_user_collections(user_id)

def delete_file(user_id: str, collection_name: str, filename: str) -> Dict[str, Any]:
    return get_kb_manager().delete_file_from_collection(user_id, collection_name, filename)

def delete_collection(user_id: str, collection_name: str) -> Dict[str, Any]:
    return get_kb_manager().delete_knowledge_base(user_id, collection_name)

def add_files(user_id: str, collection_name: str, file_paths: List[str]) -> Dict[str, Any]:
    return get_kb_manager().add_files_to_collection(user_id, collection_name, file_paths)
