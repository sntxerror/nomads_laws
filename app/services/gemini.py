import google.generativeai as genai
from ..core.config import settings
from ..core.embeddings import EmbeddingsManager

class GeminiService:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.embeddings_manager = EmbeddingsManager(settings)
        
    async def ask_legal_question(
        self,
        question: str,
        country: str = settings.DEFAULT_COUNTRY,
        law_type: str = settings.DEFAULT_LAW_TYPE,
        language: str = settings.DEFAULT_LANGUAGE
    ) -> str:
        # Retrieve relevant context from embeddings
        relevant_chunks = await self.embeddings_manager.get_relevant_context(
            query=question,
            country=country,
            law_type=law_type,
            language=language
        )
        
        context = "\n\n".join(relevant_chunks)
        
        prompt = f"""You are a Legal Assistant specializing in {country.title()} {law_type} law.
        Answer the following question in {language}.
        Base your answer only on these relevant sections of law:
        
        {context}
        
        Question: {question}"""

        response = self.model.generate_content(prompt)
        return response.text
