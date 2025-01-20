import logging
from src.database import get_all_users
from src.poloniex_api import get_ticker_data
from src.data_processing import find_significant_drops

logger = logging.getLogger(__name__)

async def job():
    """
    Основная задача для планировщика: проверяет данные тикеров, находит новые падения и восстановления, отправляет уведомления.
    """
    logger.info("Запуск задачи проверки монет.")
    ticker_data = await get_ticker_data()
    if not ticker_data:
        logger.warning("Нет данных тикеров для обработки.")
        return

    users = get_all_users()
    for user in users:
        chat_id = user['chat_id']
        thresholds = user['thresholds']
        if not thresholds:
            continue  # Пользователь не выбрал пороги

        new_drops, recovered_coins = await find_significant_drops(ticker_data, thresholds, chat_id)

        # Проверка на наличие новых падений
        if new_drops:
            logger.info(f"Найдены новые падения для пользователя {chat_id}.")

        # Проверка на восстановление
        if recovered_coins:
            logger.info(f"Найдены восстановленные монеты для пользователя {chat_id}.")
