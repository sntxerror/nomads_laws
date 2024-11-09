from telegram import Update
from telegram.ext import ContextTypes
from ..services.gemini import GeminiService

class BotHandlers:
    def __init__(self):
        self.gemini = GeminiService()
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🇬🇪 Привет! Я помогу вам разобраться с налоговым законодательством Грузии. Задавайте ваши вопросы!"
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            # Process user question through GeminiService
            answer = await self.gemini.ask_legal_question(update.message.text)
            await update.message.reply_text(answer)
        except Exception as e:
            await update.message.reply_text(f"Произошла ошибка: {str(e)}")
