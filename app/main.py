import logging
from fastapi import FastAPI
import asyncio
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from .core.config import settings
from .bot.handlers import BotHandlers
from .services.gemini import GeminiService
from .core.embeddings import EmbeddingsManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nomads Laws")
handlers = BotHandlers()
gemini_service = GeminiService()
embeddings_manager = EmbeddingsManager(settings)  # Explicitly initialize EmbeddingsManager

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    
    # Initialize Telegram bot
    try:
        logger.info("Initializing Telegram bot...")
        application = Application.builder().token(settings.TELEGRAM_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", handlers.start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.handle_message))
        
        logger.info("Telegram bot initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize Telegram bot: %s", e)

    # Load tax law document into embeddings
    try:
        logger.info("Loading tax law document for embeddings...")
        with open("app/data/georgia/tax/ru/law.txt", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Load document into EmbeddingsManager for embedding generation and upload
        await embeddings_manager.load_document(
            content=content,
            country="georgia",
            law_type="tax",
            language="ru"
        )
        logger.info("Tax law document loaded and embeddings generated successfully.")
    except FileNotFoundError:
        logger.error("Tax law document not found. Ensure 'app/data/georgia/tax/ru/law.txt' exists.")
    except Exception as e:
        logger.error("Failed to load document and generate embeddings: %s", e)
    
    # Start bot polling
    try:
        logger.info("Starting Telegram bot polling...")
        await application.initialize()
        await application.start()
        await application.run_polling()
        logger.info("Telegram bot polling started.")
    except Exception as e:
        logger.error("Failed to start Telegram bot polling: %s", e)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
