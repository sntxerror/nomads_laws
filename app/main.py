from fastapi import FastAPI
import asyncio
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from .core.config import settings
from .bot.handlers import BotHandlers
from .services.gemini import GeminiService

app = FastAPI(title="Nomads Laws")
handlers = BotHandlers()
gemini_service = GeminiService()

@app.on_event("startup")
async def startup_event():
    # Initialize Telegram bot
    application = Application.builder().token(settings.TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))
    
    # Load tax law document into embeddings
    with open("app/data/georgia/tax/ru/law.txt", 'r', encoding='utf-8') as f:
        content = f.read()
        await gemini_service.embeddings_manager.load_document(
            content=content,
            country="georgia",
            law_type="tax",
            language="ru"
        )
    
    # Start bot polling
    await application.initialize()
    await application.start()
    await application.run_polling()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
