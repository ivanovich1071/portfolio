import requests
import logging
import json
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
import os

# Загрузка переменных окружения из .env файла
load_dotenv()

POLONIEX_API_URL = os.getenv('POLONIEX_API_URL')

# Настройка логирования для этого модуля
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(RequestException),
    reraise=True
)
async def fetch_ticker_data():
    """
    Выполняет запрос к публичному API Poloniex для получения данных тикеров.
    :return: JSON-ответ от API
    :raises: RequestException, ValueError
    """
    url = POLONIEX_API_URL
    headers = {
        'User-Agent': 'Mozilla/5.0',
    }
    response = requests.get(url, headers=headers, timeout=300)
    response.raise_for_status()
    return response.json()

async def get_ticker_data():
    """
    Получает данные о тикерах с Poloniex API с обработкой ошибок и повторными попытками.
    :return: Словарь с данными тикеров или пустой словарь при ошибке.
    """
    try:
        data = await fetch_ticker_data()
        logger.info("Успешно получены данные тикеров с Poloniex.")

        # Сохранение данных в JSON файл для отладки
        with open("poloniex_ticker_data.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
        logger.info("Данные сохранены в файл poloniex_ticker_data.json.")

        return data
    except RequestException as e:
        logger.error(f"Ошибка при запросе к Poloniex API: {e}")
    except ValueError as e:
        logger.error(f"Ошибка декодирования JSON от Poloniex API: {e}")
    except Exception as e:
        logger.error(f"Неизвестная ошибка: {e}")

    return {}
