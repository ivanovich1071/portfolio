import logging
from src.config import TELEGRAM_BOT_TOKEN
from src.handlers import setup_handlers
from src.database import init_db
from src.notifier import job
import schedule
from telegram.ext import Application
import asyncio

def main():
    # Настройка логирования
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("bot.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Инициализация базы данных
    init_db()

    # Инициализация бота через Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Настройка обработчиков
    setup_handlers(application)

    # Запуск бота через polling
    application.run_polling()  # Запускаем polling напрямую
    logger.info("Бот запущен.")

def schedule_job():
    # Настройка планировщика задач
    schedule.every(15).minutes.do(job)
    while True:
        schedule.run_pending()
        asyncio.sleep(1)

if __name__ == "__main__":
    main()  # Запуск основного цикла
