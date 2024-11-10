import google.generativeai as genai
from google.cloud import aiplatform
import numpy as np
from typing import List, Dict
import logging
from google.api_core import retry

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, settings):
        self.settings = settings
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # Initialize Vertex AI
        aiplatform.init(
            project=settings.PROJECT_ID,
            location=settings.LOCATION
        )
        
        try:
            logger.info(f"Initializing Vector Search client with endpoint: {settings.VECTOR_SEARCH_ENDPOINT}")
            self.vector_search_client = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=settings.VECTOR_SEARCH_ENDPOINT,
                project=settings.PROJECT_ID,
                location=settings.LOCATION
            )
        except Exception as e:
            logger.error(f"Failed to initialize Vector Search client: {str(e)}")
            # Initialize without vector search for basic functionality
            self.vector_search_client = None

    async def load_document(self, content: str, country: str, law_type: str, language: str):
        if not self.vector_search_client:
            logger.error("Vector Search client not initialized. Cannot load document.")
            return

        try:
            chunks = self._split_into_chunks(content)
            logger.info(f"Split document into {len(chunks)} chunks")

            embeddings_data = []
            for i, chunk in enumerate(chunks):
                embedding = genai.embed_content(
                    model=self.settings.EMBEDDING_MODEL,
                    content=chunk,
                    task_type="retrieval_document"
                )["embedding"]["values"]

                embeddings_data.append({
                    "id": f"{country}-{law_type}-{language}-{i}",
                    "embedding": embedding,
                    "metadata": {
                        "country": country,
                        "law_type": law_type,
                        "language": language,
                        "text": chunk
                    }
                })

            await self._upload_embeddings(embeddings_data)
            logger.info("Successfully uploaded embeddings")

        except Exception as e:
            logger.error(f"Error in load_document: {str(e)}")
            raise

    @retry.Retry()
    async def _upload_embeddings(self, embeddings_data: List[Dict]):
        if not self.vector_search_client:
            logger.error("Vector Search client not initialized. Cannot upload embeddings.")
            return

        try:
            # Process in smaller batches to avoid quota limits
            batch_size = 100
            for i in range(0, len(embeddings_data), batch_size):
                batch = embeddings_data[i:i + batch_size]
                
                embeddings = [d["embedding"] for d in batch]
                ids = [d["id"] for d in batch]
                metadata = [d["metadata"] for d in batch]

                self.vector_search_client.upsert_embeddings(
                    embeddings=embeddings,
                    ids=ids,
                    metadata_dict=metadata
                )
                logger.info(f"Uploaded batch {i//batch_size + 1} of embeddings")
                
        except Exception as e:
            logger.error(f"Error uploading embeddings: {str(e)}")
            raise

    async def get_relevant_context(self, query: str, country: str, law_type: str, language: str, top_k: int = 3) -> List[str]:
        if not self.vector_search_client:
            logger.warning("Vector Search client not initialized. Using fallback method.")
            return []

        try:
            query_embedding = genai.embed_content(
                model=self.settings.EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"
            )["embedding"]["values"]

            response = self.vector_search_client.find_neighbors(
                embedding=query_embedding,
                num_neighbors=top_k,
                filter=f"country = '{country}' AND law_type = '{law_type}' AND language = '{language}'"
            )

            if response and hasattr(response, 'neighbors'):
                return [neighbor.metadata["text"] for neighbor in response.neighbors]
            return []

        except Exception as e:
            logger.error(f"Error in get_relevant_context: {str(e)}")
            return []

    def _split_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.settings.CHUNK_SIZE - self.settings.CHUNK_OVERLAP):
            chunk = " ".join(words[i:i + self.settings.CHUNK_SIZE])
            chunks.append(chunk)
        return chunks

    async def check_status(self):
        if not self.vector_search_client:
            return {
                "status": "error",
                "message": "Vector Search client not initialized",
                "endpoint": self.settings.VECTOR_SEARCH_ENDPOINT
            }

        try:
            dummy_embedding = [0.0] * self.settings.DIMENSION_SIZE
            response = self.vector_search_client.find_neighbors(
                embedding=dummy_embedding,
                num_neighbors=1
            )
            return {
                "status": "operational",
                "endpoint": self.settings.VECTOR_SEARCH_ENDPOINT,
                "has_data": hasattr(response, 'neighbors') and len(response.neighbors) > 0
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "endpoint": self.settings.VECTOR_SEARCH_ENDPOINT
            }