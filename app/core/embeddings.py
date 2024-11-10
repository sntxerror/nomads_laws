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
        aiplatform.init(project='nomads-laws')
        self.vector_search_client = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=settings.VECTOR_SEARCH_ENDPOINT
        )

    async def load_document(self, content: str, country: str, law_type: str, language: str):
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

            self._upload_embeddings(embeddings_data)

        except Exception as e:
            logger.error(f"Error in load_document: {str(e)}")
            raise

    def _upload_embeddings(self, embeddings_data: List[Dict]):
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
            return {"status": "error", "message": str(e)}