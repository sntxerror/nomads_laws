import google.generativeai as genai
import logging
from ..core.config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self, embeddings_manager):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.embeddings_manager = embeddings_manager
        
    async def ask_legal_question(self, question: str, country: str = settings.DEFAULT_COUNTRY,
                               law_type: str = settings.DEFAULT_LAW_TYPE, language: str = settings.DEFAULT_LANGUAGE) -> str:
        try:
            relevant_chunks = await self.embeddings_manager.get_relevant_context(
                query=question,
                country=country,
                law_type=law_type,
                language=language
            )
            
            if not relevant_chunks:
                return "Извините, я не нашел релевантной информации в законодательстве по вашему вопросу."

            context = "\n\n".join(relevant_chunks)
            
            prompt = f"""You are a Legal Assistant specializing in {country.title()} {law_type} law.
            Answer the following question in {language}.
            Base your answer only on these relevant sections of law:
            
            {context}
            
            Question: {question}"""

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error in ask_legal_question: {str(e)}")
            return "Извините, произошла ошибка при обработке вашего вопроса. Попробуйте позже."