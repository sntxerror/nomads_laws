import google.generativeai as genai
from google.cloud import aiplatform
import numpy as np
import logging
from typing import List, Dict
from ..core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingsManager:
    def __init__(self, settings):
        self.settings = settings
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.vector_search_client = aiplatform.MatchingEngineIndexEndpointClient(
            client_options={"api_endpoint": settings.VECTOR_SEARCH_ENDPOINT}
        )
        logger.info("Initialized EmbeddingsManager with Vertex AI endpoint: %s", settings.VECTOR_SEARCH_ENDPOINT)

    async def load_document(self, content: str, country: str, law_type: str, language: str):
        logger.info("Loading document for embeddings with metadata: country=%s, law_type=%s, language=%s", country, law_type, language)

        # Split the document into chunks and generate embeddings with metadata
        chunks = self._split_into_chunks(content)
        embeddings = []

        for chunk in chunks:
            # Generate an embedding for each chunk using Multilingual Model 002
            logger.info("Generating embedding for chunk: %s", chunk[:100])  # Log the first 100 characters of the chunk
            embedding = genai.embed_content(
                model=self.settings.EMBEDDING_MODEL,
                content=chunk,
                task_type="retrieval_document"
            )["embedding"]["values"]

            # Store embedding with metadata
            embeddings.append({
                "embedding": embedding,
                "metadata": {
                    "country": country,
                    "law_type": law_type,
                    "language": language,
                    "chunk_text": chunk  # Optional for reference
                }
            })

        # Upload all embeddings with metadata to Vertex AI Vector Search
        self._upload_embeddings_to_vector_search(embeddings)

    def _upload_embeddings_to_vector_search(self, embeddings: List[Dict]):
        logger.info("Uploading embeddings with metadata to Vertex AI Vector Search.")
        for embedding_data in embeddings:
            try:
                self.vector_search_client.write_index_datapoints(
                    endpoint=self.settings.VECTOR_SEARCH_ENDPOINT,
                    embeddings=embedding_data["embedding"],
                    metadata=embedding_data["metadata"]
                )
                logger.info("Uploaded embedding with metadata: %s", embedding_data["metadata"])
            except Exception as e:
                logger.error("Failed to upload embedding to Vertex AI: %s", e)

    async def get_relevant_context(
        self,
        query: str,
        country: str,
        law_type: str,
        language: str,
        top_k: int = 3
    ) -> List[str]:
        logger.info("Generating embedding for query: %s", query)
        
        # Generate an embedding for the query
        query_embedding = genai.embed_content(
            model=self.settings.EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )["embedding"]["values"]
        
        # Define metadata filters to retrieve only relevant embeddings
        metadata_filters = {
            "country": country,
            "law_type": law_type,
            "language": language
        }
        logger.info("Searching for neighbors with metadata filters: %s", metadata_filters)

        try:
            # Query Vertex AI Vector Search with filters
            response = self.vector_search_client.find_neighbors(
                endpoint=self.settings.VECTOR_SEARCH_ENDPOINT,
                embedding=query_embedding,
                metadata_filters=metadata_filters,
                neighbor_count=top_k
            )
            logger.info("Received %d neighbors from Vertex AI Vector Search.", len(response.neighbors))

            # Retrieve top-k relevant chunks from the response
            top_chunks = [neighbor.metadata["chunk_text"] for neighbor in response.neighbors]
            logger.info("Retrieved top chunks: %s", top_chunks)
            return top_chunks
        except Exception as e:
            logger.error("Failed to retrieve neighbors from Vertex AI: %s", e)
            return []

    def _split_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = [
            " ".join(words[i:i + self.settings.CHUNK_SIZE])
            for i in range(0, len(words), self.settings.CHUNK_SIZE - self.settings.CHUNK_OVERLAP)
        ]
        logger.info("Split document into %d chunks.", len(chunks))
        return chunks
