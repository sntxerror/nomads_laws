from fastapi import FastAPI, Request
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from google.cloud import logging as cloud_logging
import logging
from .core.config import settings
from .core.embeddings import EmbeddingsManager
from .services.gemini import GeminiService
from .bot.handlers import BotHandlers

# Setup logging
cloud_client = cloud_logging.Client()
cloud_client.setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="Nomads Laws")

# Initialize services
embeddings_manager = EmbeddingsManager(settings)
gemini_service = GeminiService(embeddings_manager)
bot_handlers = BotHandlers(gemini_service)

# Initialize Telegram application
telegram_app = Application.builder().token(settings.TELEGRAM_TOKEN).build()
telegram_app.add_handler(CommandHandler("start", bot_handlers.start))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_handlers.handle_message))

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting application...")
        
        # Load document
        try:
            with open("app/data/georgia/tax/ru/law.txt", 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Loaded tax law document: {len(content)} characters")
            
            success = await embeddings_manager.load_document(
                content=content,
                country=settings.DEFAULT_COUNTRY,
                law_type=settings.DEFAULT_LAW_TYPE,
                language=settings.DEFAULT_LANGUAGE
            )
            if success:
                logger.info("Document processing complete")
            else:
                logger.warning("Document processing completed with errors")
        except Exception as e:
            logger.error(f"Failed to process document: {str(e)}")
            # Continue startup even if document processing fails
        
        # Setup and verify webhook
        webhook_url = f"https://{settings.CLOUD_RUN_URL}/telegram-webhook"
        
        # Get current webhook info
        webhook_info = await telegram_app.bot.get_webhook_info()
        current_url = webhook_info.url

        if current_url != webhook_url:
            # Delete old webhook if exists
            if current_url:
                await telegram_app.bot.delete_webhook()
                logger.info(f"Deleted old webhook: {current_url}")

            # Set new webhook
            await telegram_app.bot.set_webhook(
                url=webhook_url,
                allowed_updates=['message', 'callback_query'],
                drop_pending_updates=True
            )
            logger.info(f"Set webhook URL to: {webhook_url}")
        else:
            logger.info(f"Webhook already set to correct URL: {webhook_url}")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        # Don't raise exception to allow partial functionality

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """Handle Telegram webhook requests"""
    try:
        update_data = await request.json()
        logger.info(f"Received webhook update: {update_data.get('update_id')}")
        
        update = Update.de_json(update_data, telegram_app.bot)
        await telegram_app.process_update(update)
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    try:
        webhook_info = await telegram_app.bot.get_webhook_info()
        webhook_status = {
            "url": webhook_info.url,
            "pending_updates": webhook_info.pending_update_count,
            "last_error": webhook_info.last_error_date,
            "last_error_message": webhook_info.last_error_message
        }
        
        vector_status = await embeddings_manager.check_status()
        
        return {
            "status": "healthy",
            "vector_search": vector_status,
            "telegram_webhook": webhook_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/debug")
async def debug_info():
    """Endpoint to check system status"""
    return {
        "vector_search_status": await embeddings_manager.check_status(),
        "webhook_info": await telegram_app.bot.get_webhook_info(),
        "settings": {
            "endpoint": settings.VECTOR_SEARCH_ENDPOINT,
            "model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE
        }
    }