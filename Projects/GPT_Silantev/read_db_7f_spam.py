import psycopg2
from dotenv import load_dotenv
import os
import csv
from datetime import datetime, timedelta

load_dotenv()

# Константы
CSV_PATH = "deleted_channels_log8.csv"

# Класс для управления подключением к БД
class DatabaseHandler:
    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT")
        self.database = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.cursor = self.connection.cursor()
            print("Подключение к базе данных установлено.")
        except Exception as e:
            print(f"Ошибка подключения к базе данных: {e}")
            raise

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Соединение с базой данных закрыто.")

    def execute_query(self, query, params=None, fetch=True):
        try:
            self.cursor.execute(query, params)
            if fetch:
                return self.cursor.fetchall()
            return True
        except psycopg2.Error as e:
            print(f"Ошибка выполнения запроса: {e.pgerror}")
            self.connection.rollback()
            return []

    def commit(self):
        try:
            if self.connection:
                self.connection.commit()
        except psycopg2.Error as e:
            print(f"Ошибка коммита: {e.pgerror}")
            self.connection.rollback()

# Инициализация файла CSV
def initialize_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "channel_name", "reason", "ER", "OI", "FVR", "RVR", "RR", "CRI", "CQI"
            ])
        print(f"Файл {CSV_PATH} создан.")

# Запись логов в CSV
def log_to_csv(channel_name, reason, metrics):
    with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            channel_name,
            reason,
            metrics.get("ER"),
            metrics.get("OI"),
            metrics.get("FVR"),
            metrics.get("RVR"),
            metrics.get("RR"),
            metrics.get("CRI"),
            metrics.get("CQI")
        ])
    print(f"Канал {channel_name} с причиной '{reason}' записан в лог.")

# Создание индексов
def create_indexes(db):
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_channel_name ON messages(channel_name);",
        "CREATE INDEX IF NOT EXISTS idx_date ON messages(date);"
    ]
    for index in indexes:
        db.execute_query(index, fetch=False)
    db.commit()
    print("Индексы созданы.")

# Функция для чтения структуры базы данных
def read_database_structure(db):
    query_tables = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public';
    """
    tables = db.execute_query(query_tables)
    print(f"Таблицы в базе данных:\n")
    for table in tables:
        table_name = table[0]
        print(f"Таблица: {table_name}")
        query_columns = f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table_name}';
        """
        columns = db.execute_query(query_columns)
        print("Столбцы:")
        for column in columns:
            print(f"  - {column[0]} ({column[1]}), NOT NULL: {column[2]}")
        print()

    query_channel_count = "SELECT COUNT(*) FROM channels;"
    channel_count = db.execute_query(query_channel_count)[0][0]
    print(f"Общее количество каналов в базе данных: {channel_count}\n")

# Расчет метрик
def calculate_metrics(db):
    query = """
        UPDATE channels SET
        ER = (
            SELECT COALESCE(SUM(m.reactions_count + m.shares_count + m.views_count), 0) / NULLIF(channels.subscribers, 0) * 100
            FROM messages m
            WHERE m.channel_name = channels.channel_name
        ),
        OI = 1 - (
            SELECT COALESCE(SUM(m.shares_count), 0) / NULLIF(COUNT(m.id), 0)
            FROM messages m
            WHERE m.channel_name = channels.channel_name
        ),
        FVR = (
            SELECT COALESCE(SUM(m.views_count), 0) / NULLIF(channels.subscribers, 0) * 100
            FROM messages m
            WHERE m.channel_name = channels.channel_name
        ),
        RVR = (
            SELECT COALESCE(SUM(m.reactions_count), 0) / NULLIF(SUM(m.views_count), 0) * 100
            FROM messages m
            WHERE m.channel_name = channels.channel_name
        ),
        RR = channels.subscribers / NULLIF((SELECT COALESCE(SUM(m.reactions_count), 0)
            FROM messages m
            WHERE m.channel_name = channels.channel_name), 0),
        CRI = (
            SELECT (COALESCE(SUM(m.views_count), 0) + COALESCE(SUM(m.shares_count), 0)) / NULLIF(COUNT(m.id), 0)
            FROM messages m
            WHERE m.channel_name = channels.channel_name
        ),
        CQI = 0.5 * (
            SELECT COALESCE(SUM(m.views_count), 0) / NULLIF(channels.subscribers, 0)
            FROM messages m
            WHERE m.channel_name = channels.channel_name
        ) + 0.3 * (
            SELECT COALESCE(SUM(m.reactions_count), 0) / NULLIF(channels.subscribers, 0)
            FROM messages m
            WHERE m.channel_name = channels.channel_name
        ) + 0.2 * (
            SELECT COALESCE(SUM(m.shares_count), 0) / NULLIF(channels.subscribers, 0)
            FROM messages m
            WHERE m.channel_name = channels.channel_name
        );
    """
    db.execute_query(query, fetch=False)
    db.commit()
    print("Расчет метрик завершен.")

# Получение метрик канала
def get_channel_metrics(db, channel_name):
    query = """
        SELECT ER, OI, FVR, RVR, RR, CRI, CQI
        FROM channels
        WHERE channel_name = %s;
    """
    metrics = db.execute_query(query, (channel_name,))
    if metrics:
        return {
            "ER": metrics[0][0],
            "OI": metrics[0][1],
            "FVR": metrics[0][2],
            "RVR": metrics[0][3],
            "RR": metrics[0][4],
            "CRI": metrics[0][5],
            "CQI": metrics[0][6]
        }
    return {}

# 1. Удаление дублирующихся каналов
def remove_duplicate_channels(db):
    print("Идет проверка дублирующихся каналов...")
    query = """
        SELECT channel_name, COUNT(*)
        FROM channels
        GROUP BY channel_name
        HAVING COUNT(*) > 1
    """
    duplicates = db.execute_query(query)
    for channel_name, count in duplicates:
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(channel_name, "Duplicate channel", metrics)
        print(f"Канал {channel_name} удален как дубликат.")

# 2. Удаление каналов с малым количеством подписчиков
def remove_channels_with_low_subscribers(db):
    print("Идет проверка каналов с малым количеством подписчиков...")
    query = """
        SELECT channel_name
        FROM channels
        WHERE subscribers < 1000
    """
    low_subs = db.execute_query(query)
    for channel_name, in low_subs:
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(channel_name, "Low subscribers", metrics)
        print(f"Канал {channel_name} удален из-за малого количества подписчиков.")

# 3. Удаление старых каналов
def remove_old_channels(db):
    print("Идет проверка старых каналов...")
    cutoff_date = (datetime.now() - timedelta(days=270)).strftime('%Y-%m-%d')
    query = f"""
        SELECT c.channel_name
        FROM channels c
        LEFT JOIN messages m ON c.channel_name = m.channel_name
        WHERE m.date IS NULL OR m.date < '{cutoff_date}'
    """
    old_channels = db.execute_query(query)
    for channel_name, in old_channels:
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(channel_name, "Old channel", metrics)
        print(f"Канал {channel_name} удален как старый.")

# 4. Удаление спам-каналов
def remove_spam_channels(db):
    print("Идет проверка спам-каналов...")
    """
    Логирует каналы с признаками спама в CSV.
    Условия:
      - Более 80% сообщений содержат ссылки (http, www).
      - Менее 5% сообщений содержат текст.
      - Удаляются только каналы, у которых ER < 0.5 и CQI < 0.5.
    """
    # Удаление каналов с более 80% сообщений, содержащих ссылки
    query_links = """
        SELECT c.channel_name, c.ER, c.CQI
        FROM channels c
        JOIN (
            SELECT channel_name, COUNT(*) AS total_messages,
                   SUM(CASE WHEN text LIKE '%http%' OR text LIKE '%www%' THEN 1 ELSE 0 END) AS link_count
            FROM (
                SELECT channel_name, text
                FROM messages
                WHERE date >= (SELECT MAX(date) FROM messages WHERE channel_name = c.channel_name) - INTERVAL '20 days'
                ORDER BY date DESC
                LIMIT 20
            ) sub
            GROUP BY channel_name
        ) m ON c.channel_name = m.channel_name
        WHERE link_count * 1.0 / total_messages > 0.8
        LIMIT 200
    """
    spam_channels = db.execute_query(query_links)

    for channel_name, er, cqi in spam_channels:
        if er < 0.5 and cqi < 0.5:  # Условия метрик
            metrics = get_channel_metrics(db, channel_name)
            log_to_csv(channel_name, "Spam channel (более 80% сообщений со ссылками)", metrics)
            print(f"Канал {channel_name} удален как спам-канал (более 80% сообщений со ссылками).")

    # Удаление каналов с менее 5% текстовых сообщений
    query_text = """
        SELECT c.channel_name, c.ER, c.CQI
        FROM channels c
        JOIN (
            SELECT channel_name, COUNT(*) AS total_messages,
                   SUM(CASE WHEN text IS NOT NULL AND text != '' THEN 1 ELSE 0 END) AS text_count
            FROM (
                SELECT channel_name, text
                FROM messages
                WHERE date >= (SELECT MAX(date) FROM messages WHERE channel_name = c.channel_name) - INTERVAL '20 days'
                ORDER BY date DESC
                LIMIT 20
            ) sub
            GROUP BY channel_name
        ) m ON c.channel_name = m.channel_name
        WHERE text_count * 1.0 / total_messages < 0.05
        LIMIT 200
    """
    low_text_channels = db.execute_query(query_text)

    for channel_name, er, cqi in low_text_channels:
        if er < 0.5 and cqi < 0.5:  # Условия метрик
            metrics = get_channel_metrics(db, channel_name)
            log_to_csv(channel_name, "Spam channel (менее 5% сообщений с текстом)", metrics)
            print(f"Канал {channel_name} удален как спам-канал (менее 5% сообщений с текстом).")

    print(f"Обработаны спам-каналы: {len(spam_channels) + len(low_text_channels)}")


# 5. Удаление накрученных каналов
def remove_fake_channels(db):
    print("Идет проверка накрученных каналов...")
    """
    Логирует каналы с признаками накрученности в CSV.
    Условия:
      - Высокая частота публикаций (> 25 постов в день).
      - Низкие метрики:
        - RVR < 1%
        - RR < 3%
        - ER < 0.5%
        - OI < 0.3
        - FVR < 10%
    """
    # Проверка каналов с высокой частотой публикаций
    query_high_frequency = """
        SELECT c.channel_name, COUNT(*) * 1.0 / (DATE_PART('day', MAX(m.date) - MIN(m.date)) + 1) AS freq
        FROM channels c
        JOIN (
            SELECT channel_name, date
            FROM (
                SELECT channel_name, date
                FROM messages
                WHERE date >= (SELECT MAX(date) FROM messages WHERE channel_name = c.channel_name) - INTERVAL '20 days'
                ORDER BY date DESC
                LIMIT 20
            ) sub
        ) m ON c.channel_name = m.channel_name
        GROUP BY c.channel_name
        HAVING freq > 25
        LIMIT 200
    """
    high_frequency_channels = db.execute_query(query_high_frequency)

    for channel_name, freq in high_frequency_channels:
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(
            channel_name,
            f"Fake channel (частота публикаций > 25 постов в день, частота = {freq:.2f})",
            metrics
        )
        print(f"Канал {channel_name} удален как накрученный (частота публикаций > 25 постов в день).")

    # Проверка каналов с RVR < 1%
    query_low_rvr = """
        SELECT c.channel_name, c.RVR, c.ER, c.CQI
        FROM channels c
        JOIN (
            SELECT channel_name,
                   SUM(reactions_count) AS total_reactions,
                   SUM(views_count) AS total_views
            FROM (
                SELECT channel_name, reactions_count, views_count
                FROM messages
                WHERE date >= (SELECT MAX(date) FROM messages WHERE channel_name = c.channel_name) - INTERVAL '20 days'
                ORDER BY date DESC
                LIMIT 20
            ) sub
            GROUP BY channel_name
        ) m ON c.channel_name = m.channel_name
        WHERE m.total_reactions * 1.0 / NULLIF(m.total_views, 0) * 100 < 1 AND c.ER < 0.5 AND c.CQI < 0.5
        LIMIT 200
    """
    low_rvr_channels = db.execute_query(query_low_rvr)

    for channel_name, rvr, er, cqi in low_rvr_channels:
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(
            channel_name,
            f"Fake channel (RVR < 1%, RVR = {rvr:.2f})",
            metrics
        )
        print(f"Канал {channel_name} удален как накрученный (RVR < 1%).")

    # Проверка каналов с RR < 3%
    query_low_rr = """
        SELECT c.channel_name, c.RR, c.ER, c.CQI
        FROM channels c
        JOIN (
            SELECT channel_name,
                   SUM(reactions_count) AS total_reactions
            FROM (
                SELECT channel_name, reactions_count
                FROM messages
                WHERE date >= (SELECT MAX(date) FROM messages WHERE channel_name = c.channel_name) - INTERVAL '20 days'
                ORDER BY date DESC
                LIMIT 20
            ) sub
            GROUP BY channel_name
        ) m ON c.channel_name = m.channel_name
        WHERE c.subscribers / NULLIF(m.total_reactions, 0) * 100 < 3 AND c.ER < 0.5 AND c.CQI < 0.5
        LIMIT 200
    """
    low_rr_channels = db.execute_query(query_low_rr)

    for channel_name, rr, er, cqi in low_rr_channels:
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(
            channel_name,
            f"Fake channel (RR < 3%, RR = {rr:.2f})",
            metrics
        )
        print(f"Канал {channel_name} удален как накрученный (RR < 3%).")

    # Проверка каналов с ER < 0.5%
    query_low_er = """
        SELECT c.channel_name, c.ER, c.CQI
        FROM channels c
        JOIN (
            SELECT channel_name,
                   SUM(reactions_count + shares_count + views_count) AS total_engagement
            FROM (
                SELECT channel_name, reactions_count, shares_count, views_count
                FROM messages
                WHERE date >= (SELECT MAX(date) FROM messages WHERE channel_name = c.channel_name) - INTERVAL '20 days'
                ORDER BY date DESC
                LIMIT 20
            ) sub
            GROUP BY channel_name
        ) m ON c.channel_name = m.channel_name
        WHERE m.total_engagement * 1.0 / NULLIF(c.subscribers, 0) * 100 < 0.5 AND c.CQI < 0.5
        LIMIT 200
    """
    low_er_channels = db.execute_query(query_low_er)

    for channel_name, er, cqi in low_er_channels:
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(
            channel_name,
            f"Fake channel (ER < 0.5%, ER = {er:.2f})",
            metrics
        )
        print(f"Канал {channel_name} удален как накрученный (ER < 0.5%).")

    # Проверка каналов с OI < 0.3
    query_low_oi = """
        SELECT c.channel_name, c.OI, c.ER, c.CQI
        FROM channels c
        JOIN (
            SELECT channel_name,
                   SUM(shares_count) AS total_shares,
                   COUNT(*) AS total_messages
            FROM (
                SELECT channel_name, shares_count
                FROM messages
                WHERE date >= (SELECT MAX(date) FROM messages WHERE channel_name = c.channel_name) - INTERVAL '20 days'
                ORDER BY date DESC
                LIMIT 20
            ) sub
            GROUP BY channel_name
        ) m ON c.channel_name = m.channel_name
        WHERE 1 - (m.total_shares * 1.0 / NULLIF(m.total_messages, 0)) < 0.3 AND c.ER < 0.5 AND c.CQI < 0.5
        LIMIT 200
    """
    low_oi_channels = db.execute_query(query_low_oi)

    for channel_name, oi, er, cqi in low_oi_channels:
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(
            channel_name,
            f"Fake channel (OI < 0.3, OI = {oi:.2f})",
            metrics
        )
        print(f"Канал {channel_name} удален как накрученный (OI < 0.3).")

    # Проверка каналов с FVR < 10%
    query_low_fvr = """
        SELECT c.channel_name, c.FVR, c.ER, c.CQI
        FROM channels c
        JOIN (
            SELECT channel_name,
                   SUM(views_count) AS total_views
            FROM (
                SELECT channel_name, views_count
                FROM messages
                WHERE date >= (SELECT MAX(date) FROM messages WHERE channel_name = c.channel_name) - INTERVAL '20 days'
                ORDER BY date DESC
                LIMIT 20
            ) sub
            GROUP BY channel_name
        ) m ON c.channel_name = m.channel_name
        WHERE m.total_views * 1.0 / NULLIF(c.subscribers, 0) * 100 < 10 AND c.ER < 0.5 AND c.CQI < 0.5
        LIMIT 200
    """
    low_fvr_channels = db.execute_query(query_low_fvr)

    for channel_name, fvr, er, cqi in low_fvr_channels:
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(
            channel_name,
            f"Fake channel (FVR < 10%, FVR = {fvr:.2f})",
            metrics
        )
        print(f"Канал {channel_name} удален как накрученный (FVR < 10%).")

    print("Обработаны каналы с признаками накрученности:")
    print(f"- Высокая частота публикаций: {len(high_frequency_channels)}")
    print(f"- Низкий RVR: {len(low_rvr_channels)}")
    print(f"- Низкий RR: {len(low_rr_channels)}")
    print(f"- Низкий ER: {len(low_er_channels)}")
    print(f"- Низкий OI: {len(low_oi_channels)}")
    print(f"- Низкий FVR: {len(low_fvr_channels)}")

# 6. Удаление каналов с коротким текстом
def remove_short_text_channels(db):
    print("Идет проверка каналов с коротким текстом...")
    """
    Логирует каналы, где более 50% сообщений короче 150 символов.
    Условия:
      - Более 50% сообщений короче 150 символов.
      - ER < 0.5 и CQI < 0.5.
    """
    # Запрос для каналов с короткими сообщениями
    query = """
        SELECT c.channel_name, c.ER, c.CQI
        FROM channels c
        JOIN (
            SELECT channel_name, COUNT(*) AS total_messages,
                   SUM(CASE WHEN LENGTH(text) < 150 THEN 1 ELSE 0 END) AS short_text_count
            FROM (
                SELECT channel_name, text
                FROM messages
                WHERE date >= (SELECT MAX(date) FROM messages WHERE channel_name = c.channel_name) - INTERVAL '20 days'
                ORDER BY date DESC
                LIMIT 20
            ) sub
            GROUP BY channel_name
        ) m ON c.channel_name = m.channel_name
        WHERE short_text_count * 1.0 / total_messages > 0.5
        LIMIT 200
    """
    short_text_channels = db.execute_query(query)

    for channel_name, er, cqi in short_text_channels:
        if er < 0.5 and cqi < 0.5:  # Условия для удаления
            metrics = get_channel_metrics(db, channel_name)
            log_to_csv(
                channel_name,
                "Short text channel (более 50% сообщений короче 150 символов, ER < 0.5, CQI < 0.5)",
                metrics
            )
            print(f"Канал {channel_name} удален как канал с коротким текстом.")

    print(f"Обработаны каналы с коротким текстом: {len(short_text_channels)}")

# 7. Удаление неактивных каналов
def remove_inactive_channels(db):
    print("Идет проверка неактивных каналов...")
    """
    Логирует неактивные каналы в CSV.
    Условия:
      - ER < 0.5.
      - CQI < 0.5.
    """
    # Запрос для выборки неактивных каналов
    query = """
        SELECT channel_name, ER, CQI
        FROM channels
        WHERE ER < 0.5 AND CQI < 0.5
        LIMIT 200
    """
    inactive_channels = db.execute_query(query)

    for channel_name, er, cqi in inactive_channels:
        # Логирование канала в CSV
        metrics = get_channel_metrics(db, channel_name)
        log_to_csv(
            channel_name,
            "Inactive channel (ER < 0.5, CQI < 0.5)",
            metrics
        )
        print(f"Канал {channel_name} удален как неактивный.")

    print(f"Обработаны неактивные каналы: {len(inactive_channels)}")
# Удаление спам-каналов
def remove_spam_channels():
    """
    Проверяет каналы на признаки спама, основываясь на метриках из CSV,
    и логирует результат в этот же файл.
    Условия:
      - Более 80% сообщений содержат ссылки (http, www).
      - Менее 5% сообщений содержат текст.
    """
    def safe_float(value):
        """
        Безопасное преобразование значения в float.
        Если преобразование невозможно, возвращает 0.
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0

    def read_csv_metrics(file_path):
        """
        Читает данные из CSV-файла и возвращает список словарей.
        :param file_path: Путь к CSV-файлу.
        :return: Список словарей с данными.
        """
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден.")
            return []

        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return [row for row in reader]

    def log_to_spam_csv(row_data):
        """
        Записывает данные в CSV-файл.
        :param row_data: Словарь с данными для записи.
        """
        file_exists = os.path.exists(CSV_PATH)

        with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                "channel_name", "reason", "ER", "FVR", "RR", "CRI", "CQI", "criteria"
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

    print("Чтение данных из CSV...")
    data = read_csv_metrics(CSV_PATH)
    if not data:
        print("Нет данных для обработки.")
        return

    processed_count = 0

    for row in data:
        channel_name = row.get("channel_name", "Unknown")
        er = safe_float(row.get("ER"))
        cqi = safe_float(row.get("CQI"))
        fvr = safe_float(row.get("FVR"))
        rvr = safe_float(row.get("RVR"))

        # Условия фильтрации
        if er < 0.5 or cqi < 0.3 or fvr < 10 or rvr < 1:
            processed_count += 1
            log_to_spam_csv({
                "channel_name": channel_name,
                "reason": "Удален как спам-канал",
                "ER": er,
                "FVR": fvr,
                "RR": row.get("RR", 0),
                "CRI": row.get("CRI", 0),
                "CQI": cqi,
                "criteria": "Первый этап фильтрации"
            })
            print(f"Канал {channel_name} удален как спам-канал.")

    print(f"Обработано спам-каналов: {processed_count}")
# Основная функция
def main():
    initialize_csv()
    db = DatabaseHandler()

    try:
        db.connect()
        create_indexes(db)
        read_database_structure(db)

        remove_duplicate_channels(db)  # Шаг 1
        remove_channels_with_low_subscribers(db)  # Шаг 2
      #  remove_old_channels(db)  # Шаг 3

        calculate_metrics(db)  # Расчет метрик

        # Удаление после расчета метрик
        remove_spam_channels(db)
        remove_fake_channels(db)
        remove_short_text_channels(db)
        remove_inactive_channels(db)

        print(f"Файл логов сохранен по пути: {CSV_PATH}. Скачайте файл по ссылке.")

    finally:
        db.close()

if __name__ == "__main__":
    main()
