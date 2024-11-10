from telegram import Update
from telegram.ext import ContextTypes
import logging

logger = logging.getLogger(__name__)

class BotHandlers:
    def __init__(self, gemini_service):
        self.gemini = gemini_service
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🇬🇪 Привет! Я помогу вам разобраться с налоговым законодательством Грузии. "
            "Задавайте ваши вопросы!"
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = update.effective_user.id
            question = update.message.text
            logger.info(f"Received question from user {user_id}: {question}")

            await update.message.chat.send_action("typing")
            answer = await self.gemini.ask_legal_question(question)
            await update.message.reply_text(answer)
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            await update.message.reply_text(
                "Извините, произошла ошибка при обработке вашего вопроса. "
                "Попробуйте позже или переформулируйте вопрос."
            )