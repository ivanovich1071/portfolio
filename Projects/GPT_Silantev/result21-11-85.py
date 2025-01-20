import os
import csv
import time
import numpy as np
import pickle
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# Конфигурация
CREATIVE_EMBEDDINGS_FILE = "creative_embeddings1.pkl"
CREATIVE_METADATA_FILE = "creative_metadata1.csv"
CHANNEL_EMBEDDINGS_FILE = "channel_embeddings.pkl"
CHANNEL_METADATA_FILE = "group_channel_metadata.csv"
CLUSTER_EMBEDDINGS_FILE = "cluster_embeddings.pkl"
RESULTS_FILE = "results23-11-85.csv"
RELEVANCE_THRESHOLD = 85
TIMEOUT_LIMIT = 240

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

# Вычисление косинусного сходства
def calculate_similarity(creative_embedding, channel_embedding):
    creative_embedding = normalize_embedding(creative_embedding)
    channel_embedding = normalize_embedding(channel_embedding)
    similarity_score = cosine_similarity([creative_embedding], [channel_embedding])[0][0]
    return max(0, min(similarity_score, 1))

# Загрузка эмбеддингов из pickle файла
def load_pickle_embeddings(file_path):
    """Загружает эмбеддинги из .pkl файла"""
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        print(f"Ошибка: файл {file_path} не найден.")
        return None
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {e}")
        return None

# Загрузка метаданных из CSV файла
def load_metadata(file_path):
    """Загружает метаданные из CSV файла"""
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Пропуск заголовка
            return [row for row in reader]
    except FileNotFoundError:
        print(f"Ошибка: файл {file_path} не найден.")
        return []
    except Exception as e:
        print(f"Ошибка при загрузке файла {file_path}: {e}")
        return []

def load_all_data():
    # Загрузка эмбеддингов креативов
    creative_embeddings = load_pickle_embeddings(CREATIVE_EMBEDDINGS_FILE)

    # Загрузка эмбеддингов каналов
    channel_embeddings = load_pickle_embeddings(CHANNEL_EMBEDDINGS_FILE)

    # Загрузка метаданных креативов
    creative_metadata = load_metadata(CREATIVE_METADATA_FILE)

    # Загрузка метаданных каналов
    channel_metadata = load_metadata(CHANNEL_METADATA_FILE)

    # Загрузка эмбеддингов кластеров
    try:
        with open(CLUSTER_EMBEDDINGS_FILE, 'rb') as f:
            cluster_embeddings = pickle.load(f)
    except FileNotFoundError:
        print(f"Ошибка: файл {CLUSTER_EMBEDDINGS_FILE} не найден.")
        cluster_embeddings = None
    except Exception as e:
        print(f"Ошибка при загрузке эмбеддингов кластеров: {e}")
        cluster_embeddings = None

    return creative_embeddings, channel_embeddings, creative_metadata, channel_metadata, cluster_embeddings

# Класс для взаимодействия с Llama
class LlamaChat:
    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(dotenv_path)
        self.model = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        self.timeout_limit = TIMEOUT_LIMIT  # Предел времени ожидания ответа в секундах

    def analyze_relevance(self, prompt: str, text: str, max_retries=5, retry_delay=15) -> str:
        retries = 0
        start_time = time.time()  # Засекаем время начала обработки

        while retries < max_retries:
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout_limit:
                print(f"Время обработки для запроса превысило {self.timeout_limit} секунд. Запрос пропускается.")
                return None  # Возвращаем None, если время ожидания превышено

            try:
                # Выполнение запроса к модели
                chat_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": f"{prompt}\n{text}"}]
                )
                print(f"Ответ от Llama:\n{chat_response.choices[0].message.content}\n")
                return chat_response.choices[0].message.content

            except Exception as e:
                print(f"Ошибка обработки запроса: {e}. Повтор {retries + 1} из {max_retries}.")
                time.sleep(retry_delay)
                retries += 1

        raise Exception("Не удалось получить ответ от сервера после нескольких попыток")

# Основная функция анализа
def main():
    # Загрузка всех данных
    creative_embeddings, channel_embeddings, creative_metadata, channel_metadata, cluster_embeddings = load_all_data()

    # Проверка на пустоту массивов
    if creative_embeddings is None or len(creative_embeddings) == 0 or channel_embeddings is None or len(channel_embeddings) == 0 or cluster_embeddings is None or len(cluster_embeddings) == 0:
        print("Ошибка: не удалось загрузить необходимые данные.")
        return

    # Преобразование списков в массивы NumPy
    creative_embeddings = np.array(creative_embeddings)
    channel_embeddings = np.array(channel_embeddings)
    cluster_embeddings = np.array(cluster_embeddings)

    llama_chat = LlamaChat()
    results = []

    # Универсальный промт для анализа
    prompt_text =  """
        Ты-лучший рекламный аналитик креативов и рекламных постов.
Проведи анализ креатива для каждого рекламного канала по следующему алгоритму :
1.	Тематическое соответствие и контекст
	Проверь, что контент рекламного канала и креатива связан с целевой темой рекламной кампании (например, финансы, путешествия, технологии, образование и т.д.), и оцени степень его соответствия целям кампании.
	Если кампания нацелена на долгосрочное взаимодействие с аудиторией (например, стратегический анализ или образовательный контент), убедись, что в тексте упоминаются долгосрочные цели, планирование, управление или иные аспекты, отражающие подобные намерения.
	Выдели в тексте ключевые слова и фразы, которые часто встречаются в специфике креатива (например, «долгосрочные вложения», «инновации», «образовательные ресурсы», «стратегия роста»), чтобы повысить точность соответствия.
2.	Анализ тональности и стиля контента
	Определи эмоциональный тон текста. Если креатив требует нейтрального, доверительного или аналитического подхода, проверь, что текст не содержит излишне агрессивной, развлекательной или гиперболизированной тональности.
	Обрати внимание на слова-маркеры, указывающие на рекламные намерения, такие как «самые лучшие», «выгодные предложения», «уникальные условия», — они могут снижать релевантность для кампаний, ориентированных на информативность или долгосрочную ценность.
3.	Фильтрация по ключевым словам и категориям
	Проверь, что текст содержит слова и выражения, которые указывают на соответствие рекламного канала или креатива определенному типу целевого контента:
	Для креативов, ориентированных на образовательные или информационные кампании, текст должен содержать такие термины, как «анализ», «стратегия», «долгосрочные цели».
	Если текст нацелен на краткосрочные рекламные акции, частое упоминание скидок, акций или других маркетинговых предложений («кешбэк», «выгода», «покупка») повысит релевантность.
	Отметь, если текст фокусируется на рекламных условиях и тарифах, — это сигнализирует о коммерческом характере, который может не совпадать с целями аналитического или информационного креатива.
4.	Контекстный анализ и релевантность аудитории
	Учти целевую аудиторию канала. Если текст адресован пользователям, заинтересованным в образовательных материалах, анализе рынка или долгосрочных инвестициях, это повышает релевантность для таких креативов.
	Оцени, ориентирован ли текст на аудиторию, заинтересованную в быстрых выгодах или скидках — такой контент может быть менее релевантен для кампаний, связанных с долгосрочными стратегическими темами.
5.	Проверка на наличие отраслевой лексики
	Убедись, что текст использует профессиональную лексику, характерную для конкретной отрасли (например, финансовый анализ, технологии, медиа). Профессиональные термины, такие как «доходность», «прогнозирование», «оптимизация затрат», «диверсификация» для финансовых креативов, повысят релевантность.
	Избегай упрощенной или обиходной лексики, если креатив требует аналитического и профессионального подхода.
6.	Оценка структуры и формата текста
	Проверь, насколько текст структурирован: аналитические обзоры, разбивка на секции, наличие таблиц или ссылок на дополнительные источники повышают релевантность для сложных рекламных кампаний.
	Если текст имеет упрощенный формат (краткие рекламные объявления, блоки акций), это может снизить релевантность для кампаний, ориентированных на глубокий анализ.
7.	Метрика на основе эмбеддингов и калибровка
	Используй обученные эмбеддинги, ориентированные на целевую тему (например, финансы, технологии), и подсчитывай косинусное расстояние между векторами креатива и канала.
	Применяй пороговые значения для фильтрации по релевантности, если косинусное расстояние показывает слабое совпадение. Для более точного соответствия добавь весовые коэффициенты, отражающие профессиональную лексику и тональность.

8.	Финальная оценка и рекомендации по улучшению
	Заверши анализ, присвоив тексту оценку релевантности по шкале от -2 до 2, с учетом следующих критериев:
	2 — Полное соответствие (тематика, стиль, лексика и целевая аудитория совпадают).
	1 — Умеренное соответствие (частичное совпадение по теме, но возможны отличия в тоне или формате).
	0 — Нейтральное совпадение (тематика близка, но цели креатива не отражены).
	-1 — Слабое соответствие (заметные различия в теме и стиле).
	-2 — Полное несоответствие (контент противоречит теме или целям креатива).

    """

    # Проход по каждому креативу
    for creative_index, creative_embedding in enumerate(creative_embeddings):
        creative_row = creative_metadata[creative_index]
        creative_text = creative_row[1] if len(creative_row) > 1 else "Текст креатива отсутствует"

        print(f"Текущий креатив (индекс {creative_index}): {creative_text}")

        # Проход по каждому кластеру
        for cluster_index, cluster_embedding in enumerate(cluster_embeddings):
            # Сравнение эмбеддинга креатива и кластера
            cluster_similarity = calculate_similarity(creative_embedding, cluster_embedding)
            cluster_relevance_score = cluster_similarity * 100

            print(f"Креатив {creative_index} -> Кластер {cluster_index}: Релевантность {cluster_relevance_score:.2f}%")

            # Если релевантность ниже порога, пропускаем кластер
            if cluster_relevance_score < RELEVANCE_THRESHOLD:
                print(f"Кластер {cluster_index} отклонен из-за низкой релевантности.")
                continue

            print(f"Кластер {cluster_index} прошел порог релевантности. Проверяем каналы...")

            # Проход по каналам внутри текущего кластера
            for channel_index, channel_row in enumerate(channel_metadata):
                # Проверяем, к какому кластеру принадлежит канал
                channel_cluster = channel_row[7] if len(channel_row) > 7 else None  # Используем индекс 7 для Cluster
                if channel_cluster != str(cluster_index):
                    continue

                # Получаем эмбеддинг канала
                channel_embedding = channel_embeddings[channel_index]

                category = channel_row[0] if len(channel_row) > 0 else "N/A"
                channel_name = channel_row[1] if len(channel_row) > 1 else "N/A"
                subscribers = channel_row[2] if len(channel_row) > 2 else "N/A"
                title = channel_row[3] if len(channel_row) > 3 else "N/A"
                description = channel_row[4] if len(channel_row) > 4 else "N/A"
                combined_text = channel_row[5] if len(channel_row) > 5 else "N/A"
                audience_profiles = channel_row[6] if len(channel_row) > 6 else "N/A"

                # Вывод текущего канала
                print(f"Канал (индекс {channel_index}): {channel_name}")

                # Вычисление релевантности
                similarity_score = calculate_similarity(creative_embedding, channel_embedding)
                relevance_score = similarity_score * 100
                if relevance_score < RELEVANCE_THRESHOLD:
                    print(f"Канал {channel_name} отклонен из-за низкой релевантности.")
                    continue

                # Дополнительный анализ релевантности с Llama
                analysis = llama_chat.analyze_relevance(prompt_text, creative_text)

                # Добавление результата
                results.append({
                    "Текст креатива": creative_text,
                    "Cluster": cluster_index,
                    "Категория": category,
                    "Channel Name": channel_name,
                    "Subscribers": subscribers,
                    "Название канала": title,
                    "Описание": description,
                    "Posts": combined_text,  # Используем Combined_Text для Posts
                    "Audience Profiles": audience_profiles,
                    "Баллы релевантности": round(relevance_score, 2),
                    "Оценка релевантности по шкале (-2,-1,0,1,2)": analysis
                })

    # Сохранение результатов
    with open(RESULTS_FILE, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "Текст креатива", "Cluster", "Категория", "Channel Name",
            "Subscribers", "Название канала", "Описание", "Posts",
            "Audience Profiles", "Баллы релевантности", "Оценка релевантности по шкале (-2,-1,0,1,2)"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Результаты сохранены в файл {RESULTS_FILE}")

if __name__ == "__main__":
    main()
