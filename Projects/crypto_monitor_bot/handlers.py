import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext, CommandHandler, CallbackQueryHandler, Application
from src.database import add_user, update_user_thresholds, get_user_thresholds
from src.poloniex_api import get_ticker_data
from src.data_processing import find_significant_drops
from src.telegram_bot import send_telegram_message
from src.notifier import job  # Для запуска планировщика

logger = logging.getLogger(__name__)

# Настройка обработчиков
def setup_handlers(application: Application):
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(CommandHandler("report", report))

# Обработчик команды /start
async def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    add_user(chat_id)
    keyboard = [
        [InlineKeyboardButton(">50%", callback_data='50')],
        [InlineKeyboardButton(">30%", callback_data='30')],
        [InlineKeyboardButton(">10%", callback_data='10')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Добро пожаловать! Выберите диапазоны колебаний котировок, по которым вы хотите получать отчёты:",
        reply_markup=reply_markup
    )

# Обработчик нажатия кнопок
async def button(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    threshold = float(query.data)
    update_user_thresholds(chat_id, threshold)

    # Получаем обновленные пороги
    thresholds = get_user_thresholds(chat_id)
    threshold_text = ", ".join([f">{t}%" for t in thresholds]) if thresholds else "не выбраны"
    await query.edit_message_text(text=f"Вы выбрали диапазоны изменения: {threshold_text}.")

# Обработчик команды /report
async def report(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    thresholds = get_user_thresholds(chat_id)
    if not thresholds:
        await update.message.reply_text("Вы не выбрали ни один диапазон. Пожалуйста, используйте /start для выбора.")
        return

    ticker_data = await get_ticker_data()
    if not ticker_data:
        await update.message.reply_text("Не удалось получить данные о монетах. Попробуйте позже.")
        return

    new_drops, recovered_coins = await find_significant_drops(ticker_data, thresholds, chat_id)
    message = ""

    if new_drops:
        message += "📉 Найдены монеты с резким падением:\n"
        for threshold, coins in new_drops.items():
            message += f"\n🔹 Изменение более чем на {threshold}%:\n"
            for coin in coins:
                message += f"• {coin['name']} - Объем торгов: {coin['volume']}\n"

    if recovered_coins:
        message += "\n📈 Монеты восстановились до исходного уровня или выше:\n"
        for coin in recovered_coins:
            message += f"• {coin}\n"

    if not message:
        message = "Нет монет, соответствующих выбранным диапазонам."

    await update.message.reply_text(message)
