import google.generativeai as genai
from google.cloud import aiplatform
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, settings):
        self.settings = settings
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        aiplatform.init(project=settings.PROJECT_ID, location=settings.LOCATION)
        
        try:
            logger.info(f"Initializing Vector Search client with endpoint: {settings.VECTOR_SEARCH_ENDPOINT}")
            self.vector_search_client = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=settings.VECTOR_SEARCH_ENDPOINT
            )
        except Exception as e:
            logger.error(f"Failed to initialize Vector Search client: {str(e)}")
            self.vector_search_client = None

    async def load_document(self, content: str, country: str, law_type: str, language: str):
        try:
            # Split into smaller chunks
            chunks = self._split_into_chunks(content)
            logger.info(f"Split document into {len(chunks)} chunks")

            # Process chunks in smaller batches
            batch_size = 5  # Process 5 chunks at a time
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {len(chunks)//batch_size + 1}")
                
                embeddings_data = []
                for j, chunk in enumerate(batch):
                    try:
                        # Limit chunk size if needed
                        if len(chunk.encode('utf-8')) > 8000:  # Leave some buffer
                            chunk = chunk[:4000]  # Approximate size in bytes
                        
                        embedding = genai.embed_content(
                            model=self.settings.EMBEDDING_MODEL,
                            content=chunk,
                            task_type="retrieval_document"
                        )["embedding"]["values"]

                        embeddings_data.append({
                            "id": f"{country}-{law_type}-{language}-{i+j}",
                            "embedding": embedding,
                            "metadata": {
                                "country": country,
                                "law_type": law_type,
                                "language": language,
                                "text": chunk
                            }
                        })
                    except Exception as e:
                        logger.error(f"Error processing chunk {i+j}: {str(e)}")
                        continue

                # Upload batch if we have any successful embeddings
                if embeddings_data:
                    await self._upload_embeddings(embeddings_data)
                    logger.info(f"Successfully uploaded batch {i//batch_size + 1}")

            logger.info("Document processing complete")
            return True

        except Exception as e:
            logger.error(f"Error in load_document: {str(e)}")
            return False

    async def _upload_embeddings(self, embeddings_data: List[Dict]):
        if not self.vector_search_client:
            logger.error("Vector Search client not initialized")
            return

        try:
            embeddings = [d["embedding"] for d in embeddings_data]
            ids = [d["id"] for d in embeddings_data]
            metadata = [d["metadata"] for d in embeddings_data]

            self.vector_search_client.upsert_embeddings(
                embeddings=embeddings,
                ids=ids,
                metadata_dict=metadata
            )
        except Exception as e:
            logger.error(f"Error uploading embeddings: {str(e)}")
            raise

    async def get_relevant_context(self, query: str, country: str, law_type: str, language: str, top_k: int = 3) -> List[str]:
        if not self.vector_search_client:
            logger.warning("Vector Search client not initialized")
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
        # Smaller chunk size to ensure we stay under payload limits
        chunk_size = 500  # words per chunk
        overlap = 50      # words overlap
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks

    async def check_status(self):
        try:
            if not self.vector_search_client:
                return {
                    "status": "error",
                    "message": "Vector Search client not initialized",
                    "endpoint": self.settings.VECTOR_SEARCH_ENDPOINT
                }

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