from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from typing import List, Dict, Optional, Any
import numpy as np
import asyncio
import logging
from time import sleep

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, settings):
        """Initialize the EmbeddingsManager with configuration settings."""
        self.settings = settings
        
        # Initialize Vertex AI with project settings
        aiplatform.init(project=settings.PROJECT_ID, location=settings.LOCATION)
        
        try:
            logger.info(f"Initializing Vector Search client with endpoint: {settings.VECTOR_SEARCH_ENDPOINT}")
            self.vector_search_client = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=settings.VECTOR_SEARCH_ENDPOINT
            )
        except Exception as e:
            logger.error(f"Failed to initialize Vector Search client: {str(e)}")
            self.vector_search_client = None
        
        # Initialize the embedding model
        try:
            self.embedding_model = TextEmbeddingModel.from_pretrained(settings.EMBEDDING_MODEL)
            logger.info(f"Successfully initialized embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            self.embedding_model = None

    async def load_document(self, content: str, country: str, law_type: str, language: str) -> bool:
        """
        Process and load a document into the vector store.
        
        Args:
            content: The text content to process
            country: Country code (e.g., 'georgia')
            law_type: Type of law (e.g., 'tax')
            language: Language code (e.g., 'ru')
            
        Returns:
            bool: Success status of the operation
        """
        if not self.embedding_model:
            logger.error("Embedding model not initialized")
            return False

        try:
            # Split into chunks
            chunks = self._split_into_chunks(content)
            logger.info(f"Split document into {len(chunks)} chunks")

            # Process chunks in batches
            batch_size = 5  # Process 5 chunks at a time
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(chunks), batch_size):
                batch = chunks[batch_idx:batch_idx + batch_size]
                logger.info(f"Processing batch {batch_idx//batch_size + 1} of {total_batches}")
                
                embeddings_data = []
                for chunk_idx, chunk in enumerate(batch):
                    try:
                        # Generate embedding for chunk
                        embedding = await self._generate_embedding(
                            text=chunk,
                            title=f"{country}-{law_type}-{language}",
                            is_document=True
                        )
                        
                        if embedding:
                            embeddings_data.append({
                                "id": f"{country}-{law_type}-{language}-{batch_idx+chunk_idx}",
                                "embedding": embedding,
                                "metadata": {
                                    "country": country,
                                    "law_type": law_type,
                                    "language": language,
                                    "text": chunk
                                }
                            })
                    except Exception as e:
                        logger.error(f"Error processing chunk {batch_idx+chunk_idx}: {str(e)}")
                        continue

                # Upload batch if we have any successful embeddings
                if embeddings_data:
                    success = await self._upload_embeddings(embeddings_data)
                    if success:
                        logger.info(f"Successfully uploaded batch {batch_idx//batch_size + 1}")
                    else:
                        logger.error(f"Failed to upload batch {batch_idx//batch_size + 1}")
                
                # Rate limiting
                await asyncio.sleep(1)

            logger.info("Document processing complete")
            return True

        except Exception as e:
            logger.error(f"Error in load_document: {str(e)}")
            return False

    async def _generate_embedding(self, text: str, title: Optional[str] = None, is_document: bool = False) -> Optional[List[float]]:
        """
        Generate embedding for a piece of text.
        
        Args:
            text: Text to generate embedding for
            title: Optional title for document context
            is_document: Whether this is a document (vs query)
            
        Returns:
            List[float]: Embedding vector or None if generation fails
        """
        try:
            # Prepare input
            task_type = "RETRIEVAL_DOCUMENT" if is_document else "RETRIEVAL_QUERY"
            
            # Truncate if needed
            if len(text.encode('utf-8')) > 8000:
                text = text[:4000]  # Approximate size in bytes
            
            # Create embedding input
            embedding_input = TextEmbeddingInput(
                text=text,
                task_type=task_type,
                title=title if is_document else None
            )
            
            # Get embedding
            embeddings = self.embedding_model.get_embeddings([embedding_input])
            return embeddings[0].values if embeddings else None

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    async def _upload_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """
        Upload embeddings to Vector Search.
        
        Args:
            embeddings_data: List of embedding records to upload
            
        Returns:
            bool: Success status
        """
        if not self.vector_search_client:
            logger.error("Vector Search client not initialized")
            return False

        try:
            embeddings = [d["embedding"] for d in embeddings_data]
            ids = [d["id"] for d in embeddings_data]
            metadata = [d["metadata"] for d in embeddings_data]

            self.vector_search_client.upsert_embeddings(
                embeddings=embeddings,
                ids=ids,
                metadata_dict=metadata
            )
            return True
            
        except Exception as e:
            logger.error(f"Error uploading embeddings: {str(e)}")
            return False

    async def get_relevant_context(
        self,
        query: str,
        country: str,
        law_type: str,
        language: str,
        top_k: int = 3
    ) -> List[str]:
        """
        Get relevant document chunks for a query.
        
        Args:
            query: Search query
            country: Country filter
            law_type: Law type filter
            language: Language filter
            top_k: Number of results to return
            
        Returns:
            List[str]: List of relevant text chunks
        """
        if not all([self.vector_search_client, self.embedding_model]):
            logger.warning("Services not fully initialized")
            return []

        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(
                text=query,
                is_document=False
            )
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []

            # Search for similar vectors
            response = self.vector_search_client.find_neighbors(
                embedding=query_embedding,
                num_neighbors=top_k,
                filter=f"country = '{country}' AND law_type = '{law_type}' AND language = '{language}'"
            )

            # Extract and return relevant text chunks
            if response and hasattr(response, 'neighbors'):
                return [n.metadata["text"] for n in response.neighbors]
            return []

        except Exception as e:
            logger.error(f"Error in get_relevant_context: {str(e)}")
            return []

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into manageable chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        
        # Use smaller chunks to ensure we stay under token limits
        chunk_size = min(self.settings.CHUNK_SIZE, 500)  # words per chunk
        overlap = min(self.settings.CHUNK_OVERLAP, 50)   # words overlap
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks

    async def check_status(self) -> Dict[str, Any]:
        """
        Check the status of embedding and vector search services.
        
        Returns:
            Dict[str, Any]: Status information
        """
        try:
            status = {
                "embedding_model": {
                    "initialized": self.embedding_model is not None,
                    "model_name": self.settings.EMBEDDING_MODEL
                },
                "vector_search": {
                    "initialized": self.vector_search_client is not None,
                    "endpoint": self.settings.VECTOR_SEARCH_ENDPOINT
                }
            }

            # Test vector search if initialized
            if self.vector_search_client:
                try:
                    # Create a dummy embedding for testing
                    dummy_embedding = [0.0] * self.settings.DIMENSION_SIZE
                    response = self.vector_search_client.find_neighbors(
                        embedding=dummy_embedding,
                        num_neighbors=1
                    )
                    status["vector_search"]["operational"] = True
                    status["vector_search"]["has_data"] = (
                        hasattr(response, 'neighbors') and 
                        len(response.neighbors) > 0
                    )
                except Exception as e:
                    status["vector_search"]["operational"] = False
                    status["vector_search"]["error"] = str(e)

            return status

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "endpoint": self.settings.VECTOR_SEARCH_ENDPOINT
            }