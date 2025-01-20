import logging
from src.database import get_dropped_coins, add_dropped_coin, remove_dropped_coin
from src.telegram_bot import send_telegram_message

logger = logging.getLogger(__name__)

async def find_significant_drops(ticker_data, user_thresholds, chat_id):
    """
    Найти монеты с падением на заданные пороги и отправить уведомления.
    :param ticker_data: Данные тикера Poloniex
    :param user_thresholds: Список порогов (например, [50, 30, 10])
    :param chat_id: ID пользователя
    :return: два словаря: new_drops и recovered_coins
    """
    new_drops = {}  # Монеты, которые упали на определенный процент
    recovered_coins = []  # Монеты, которые восстановились

    # Получаем список ранее отслеживаемых упавших монет для пользователя
    dropped_coins = get_dropped_coins(chat_id)
    dropped_coins_dict = {coin.coin: coin.threshold for coin in dropped_coins}

    # Обрабатываем полученные данные тикеров
    for data in ticker_data:
        try:
            coin = data.get("symbol", "")  # Обрабатываем монету
            percent_change = float(data.get("dailyChange", "0")) * 100  # Изменение за 24 часа в %
            volume = data.get("amount", "0")  # Объем торгов

            # Проверка на падение монет
            for threshold in user_thresholds:
                if percent_change <= -threshold:
                    # Добавляем монету в отслеживаемые падения, если она новая или ее порог изменился
                    if coin not in dropped_coins_dict or dropped_coins_dict[coin] < threshold:
                        add_dropped_coin(chat_id, coin, threshold)
                        new_drops.setdefault(threshold, []).append({
                            "name": coin,
                            "volume": volume
                        })

                    # Отправляем уведомление о новом падении
                    message = f"📉 Монета {coin} упала более чем на {threshold}%.\nОбъем торгов: {volume}"
                    await send_telegram_message(chat_id, message)
                    logger.info(f"Отправлено уведомление о падении {coin} на {threshold}% для пользователя {chat_id}.")
                    break  # Выходим из цикла после добавления

            # Проверка на восстановление монет
            if percent_change >= 0:
                if coin in dropped_coins_dict:
                    remove_dropped_coin(chat_id, coin)
                    recovered_coins.append(coin)
                    logger.info(f"Монета {coin} восстановилась до исходного уровня.")
        except ValueError:
            logger.error(f"Ошибка обработки данных для монеты {coin}.")
            continue

    return new_drops, recovered_coins
