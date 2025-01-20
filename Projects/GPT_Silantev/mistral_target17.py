import os
import time
import sqlite3
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# Конфигурация
DB_PATH = "C:/Users/Dell/Documents/GitHub/channels_for_UII.db"
CHANNEL_METADATA_FILE = "group_channel_metadata.csv"
CHANNEL_EMBEDDINGS_FILE = "channel_embeddings.pkl"
CLUSTER_EMBEDDINGS_FILE = "cluster_embeddings.pkl"
NUM_CLUSTERS = 8
TIMEOUT_LIMIT = 60
MAX_TOKENS = 16384


class MistralChat:
    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(dotenv_path)
        self.model = "mistral-large-latest"
        self.embedding_model = "mistral-embed"
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def get_embeddings(self, texts):
        try:
            clean_texts = [text for text in texts if isinstance(text, str) and text.strip()]
            response = self.client.embeddings.create(model=self.embedding_model, inputs=clean_texts)
            return np.array([data.embedding for data in response.data])
        except Exception as e:
            print(f"Ошибка при получении эмбеддингов: {e}")
            return None

    def get_audience_analysis(self, prompt: str, text: str, max_retries=5, retry_delay=15) -> str:
        retries = 0
        start_time = time.time()
        while retries < max_retries:
            elapsed_time = time.time() - start_time
            if elapsed_time > TIMEOUT_LIMIT:
                print(f"Время обработки для канала превысило {TIMEOUT_LIMIT} секунд. Канал пропускается.")
                return None
            try:
                chat_response = self.client.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": f"PROMPT {prompt}\n{text}"}]
                )
                return chat_response.choices[0].message.content
            except Exception as e:
                retries += 1
                print(f"Ошибка: {e}, попытка {retries}")
                time.sleep(retry_delay)
        raise Exception("Не удалось получить ответ от сервера после нескольких попыток")


# Подключение к базе данных
def connect_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        print("Подключено к базе данных.")
        return conn
    except sqlite3.Error as e:
        print(f"Ошибка подключения: {e}")
        return None


# Извлечение каналов из базы данных
def fetch_channels(conn):
    query = """
        SELECT category, channel_name, subscribers, title, description, Combined_Text
        FROM channels
        WHERE Combined_Text IS NOT NULL
    """
    cursor = conn.cursor()
    cursor.execute(query)
    return cursor.fetchall()


# Создание профилей ЦА
def create_audience_profiles(df_channels):
    mistral_chat = MistralChat()
    prompt_text = """
                Ты — аналитический ИИ, разработанный для анализа целевой аудитории (ЦА) каналов, используя загруженные данные и заданные инструкции. Вам нужно на основе предоставленного алгоритма анализа ЦА и таблицы с примерами создавать три профиля целевой аудитории для каждого рекламного канала. Эти профили должны включать параметры возраст, социальный статус, потребности и интересы, как это описано в алгоритме.

        Шаги для анализа
        Изучение содержимого канала:
        1. Оцените содержание последних сообщений и ключевые темы.
        2. Обратите внимание на язык и стиль общения, чтобы определить возраст, статус и возможные интересы аудитории.
        3. Обрати внимание на все косвенные признаки содержимого канал, такие как посты и репосты, что бы оценить вовлеченность и заинтересованность ЦА
        Идентификация демографических характеристик:
        1. Возраст: Определяйте возрастную группу (например, 18–25, 25–35  и т.д. в плоть до 70-90), основываясь на темах (образование, работа, инвестиции ,отдых, .и пр.) и языке.
        2. Социальный статус: Учитывайте упоминания или косвенные признаки профессионального уровня аудитории (например, студенты, школьники,молодые специалисты, руководители,менеджеры,инженерно-технические работники и так далее).
        3. Семейное положение(женат/холост/в отношениях), наличие детей
        Определение потребностей и интересов:
        1. Потребности выявляйте на основе прямых запросов и проблем, описанных в сообщениях (например, советы по инвестициям могут указывать на интерес к финансовому благополучию).
        2. Интересы определяйте по типу публикуемого контента, хештегам и категориям (например, если часто упоминаются путешествия или кулинария).

        Составление профилей ЦА:
        1. На основе шагов 1–3 сформируйте три профиля целевой аудитории, отражающих разные сегменты подписчиков канала.
        2. постарайся, чтобы профили были максимально точными и соответствовали стандарту.
        3. Формат вывода профилей: Для каждого рекламного канала создавайте три профиля ЦА и для каждого профиля сделай по-строчный вывод с параметрами:
        - Возраст: укажите диапазон возрастов.
        - Социальный статус: определите профессию, позицию или жизненный этап.
        - Потребности: выделите основные потребности этой ЦА (например, финансовое благополучие, карьерный рост).
        - Интересы: укажите ключевые интересы (например, активный отдых, инвестиции).
        *Итоговый вывод :На основании проведенного анализа выведи отдельной строкой профиль наиболее вероятной ЦА.
                """
    profiles = []

    for _, row in df_channels.iterrows():
        print(f"Создание профиля ЦА для канала {row['Channel Name']}...")
        audience_profile = mistral_chat.get_audience_analysis(prompt_text, row["Combined_Text"])
        profiles.append(audience_profile or "Профиль не создан")

    df_channels["Audience Profiles"] = profiles


# Создание и сохранение эмбеддингов каналов с обработкой ошибок
def create_and_save_channel_embeddings(df_channels):
    mistral_chat = MistralChat()
    combined_texts = (df_channels["Combined_Text"] + " " + df_channels["Audience Profiles"]).tolist()

    print("Генерация эмбеддингов для каналов...")
    channel_embeddings = []
    valid_indices = []  # Для синхронизации эмбеддингов с df_channels

    for idx, text in enumerate(combined_texts):
        channel_name = df_channels.loc[idx, "Channel Name"]
        print(f"Начато создание эмбеддинга для канала: {channel_name}")

        if not isinstance(text, str) or not text.strip():
            print(f"Пропуск пустого текста для канала {channel_name}.")
            continue

        start_time = time.time()  # Запоминаем время начала обработки

        try:
            if len(text.split()) > MAX_TOKENS:
                print(f"Пропуск текста для канала {channel_name} из-за превышения лимита токенов.")
                text = " ".join(text.split()[:MAX_TOKENS])  # Обрезаем текст до лимита

            # Получение эмбеддинга
            embedding = mistral_chat.get_embeddings([text])
            if embedding is not None:
                channel_embeddings.append(embedding[0])
                valid_indices.append(idx)

        except Exception as e:
            print(f"Ошибка при обработке канала {channel_name}: {e}")
            continue

        # Проверяем время обработки
        elapsed_time = time.time() - start_time
        if elapsed_time > 300:  # 5 минут = 300 секунд
            print(f"Предупреждение: Время обработки канала {channel_name} превысило 5 минут.")

        # Ограничение скорости запросов
        time.sleep(5)

    if not channel_embeddings:
        print("Эмбеддинги не были сгенерированы. Проверьте источник данных.")
        return None, None

    # Синхронизация с исходными данными
    df_channels = df_channels.iloc[valid_indices].reset_index(drop=True)

    # Сохранение эмбеддингов в файл
    with open(CHANNEL_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(channel_embeddings, f)
    print(f"Эмбеддинги каналов сохранены в файл {CHANNEL_EMBEDDINGS_FILE}.")

    return np.array(channel_embeddings), df_channels




def cluster_channels(channel_embeddings, df_channels):
    print("Кластеризация каналов...")
    if channel_embeddings.shape[0] != len(df_channels):
        raise ValueError(f"Количество эмбеддингов ({channel_embeddings.shape[0]}) не совпадает с количеством каналов ({len(df_channels)}). Проверьте входные данные.")

    try:
        normalized_embeddings = normalize(channel_embeddings, axis=1)

        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
        kmeans_labels = kmeans.fit_predict(normalized_embeddings)
        df_channels["Cluster"] = kmeans_labels

        # Сохранение эмбеддингов кластеров
        cluster_embeddings = kmeans.cluster_centers_
        with open(CLUSTER_EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(cluster_embeddings, f)
        print(f"Эмбеддинги кластеров сохранены в файл {CLUSTER_EMBEDDINGS_FILE}.")

    except Exception as e:
        print(f"Ошибка во время кластеризации: {e}")
        raise


# Основной процесс
def main():

    conn = connect_db(DB_PATH)
    if not conn:
        return

    try:
        # Извлечение данных о каналах
        channels_data = fetch_channels(conn)
        if not channels_data:
            print("Ошибка: Нет данных для обработки в базе данных.")
            return

        # Создание DataFrame из данных каналов
        df_channels = pd.DataFrame(channels_data, columns=[
            "Category", "Channel Name", "Subscribers", "Title", "Description", "Combined_Text"
        ])
        print(f"Извлечено {len(df_channels)} каналов для обработки.")

        # Создание профилей ЦА
        print("Начинается создание профилей ЦА...")
        create_audience_profiles(df_channels)
        print("Профили ЦА успешно созданы.")

        # Генерация и сохранение эмбеддингов каналов
        print("Начинается генерация эмбеддингов каналов...")
        channel_embeddings, df_channels = create_and_save_channel_embeddings(df_channels)
        print(f"Генерация эмбеддингов завершена. Эмбеддинги сохранены в {CHANNEL_EMBEDDINGS_FILE}.")

        # Кластеризация каналов
        print("Начинается кластеризация каналов...")
        cluster_channels(channel_embeddings, df_channels)
        print(f"Кластеризация завершена. Результаты сохранены в {CLUSTER_EMBEDDINGS_FILE}.")

        # Сохранение метаданных каналов
        print("Сохранение метаданных каналов...")
        df_channels.to_csv(CHANNEL_METADATA_FILE, index=False)
        print(f"Метаданные каналов сохранены в файл {CHANNEL_METADATA_FILE}.")

    except Exception as e:
        print(f"Ошибка выполнения: {e}")
    finally:
        # Закрытие соединения с базой данных
        conn.close()
        print("Соединение с базой данных закрыто.")


if __name__ == "__main__":
    main()
