# Custom Russian Tokenizer — Docker и JupyterLab

Окружение для учебного проекта на основе Docker-конфигурации `DZ_1_fundament_NLP`.
Python 3.11, JupyterLab, Hugging Face Datasets, Tokenizers и Transformers.
Веса нейросети и PyTorch для обучения и сравнения токенизаторов не нужны.

## Запуск

Нужен запущенный Docker Desktop. В папке проекта:

```bash
# 1. Собрать и запустить окружение
docker compose up -d --build

# 2. Скачать и проверить корпус
docker compose exec notebook python scripts/download_data.py

# 3. Получить токен JupyterLab
docker compose logs notebook
```

Откройте http://localhost:8889/lab и введите токен из логов Jupyter.
Также можно открыть ссылку из логов, заменив порт `8888` на `8889`.
Аутентификация токеном включена; порт доступен только с этого компьютера.

В JupyterLab создайте notebook с ядром **Python 3 (ipykernel)**.
Папка проекта подключена к `/workspace`: сохранённые notebooks и модели остаются
на компьютере после остановки или пересоздания контейнера.

Проверка библиотек в ячейке:

```python
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer

print("Окружение готово")
```

Корпус загружается отдельной командой выше. Для первой загрузки нужен интернет.
Готовый токенизатор загружается отдельно в notebook:

```python
baseline = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
```

Кэш Hugging Face находится в отдельном Docker volume и сохраняется при обычном
`docker compose down`. Датасет и готовая модель не загружаются при сборке образа.

## Управление

```bash
# Состояние и готовность сервиса
docker compose ps

# Логи сервера и токен доступа
docker compose logs notebook

# Остановить окружение
docker compose down

# После изменения requirements.txt пересобрать окружение
docker compose up -d --build

# Зафиксировать фактически установленные версии для отчёта
docker compose exec -T notebook pip freeze > requirements.lock.txt
```

Для другого порта скопируйте `.env.example` в `.env` и измените `JUPYTER_PORT`.
По умолчанию используется `8889`, чтобы не пересекаться с портом `8888`
предыдущего проекта. Зависимости заданы диапазонами; фактические версии можно
сохранить командой выше после успешной сборки.

## Локальный корпус OPUS Books

Команда загрузки (выполняется из папки проекта при запущенном контейнере):

```bash
docker compose exec notebook python scripts/download_data.py
```

Скрипт скачивает только Parquet-файл `en-ru` через `huggingface_hub`.
Установка Git LFS и клонирование репозитория не требуются.
Версия корпуса зафиксирована коммитом
`1f9f6191d0e91a3c539c2595e2fe48fc1420de9b`.

Файл сохраняется в `data/opus_books/en-ru/`. Вся папка `data/` исключена
из Git проекта и из контекста сборки Docker; преподавателю передаются
скрипт и конфигурация, а данные он скачивает той же командой.
В контейнере папка доступна через `/workspace/data/opus_books`.

Повторный запуск проверяет SHA-256, читаемость Parquet, поле `translation.ru`
и непустое содержимое. Если файл корректен, скачивание пропускается и интернет
не нужен. Повреждённый файл или LFS-указатель скачивается заново.
После успешной проверки выводятся путь и количество записей (17 496).
Ошибка загрузки или проверки завершает команду с ненулевым кодом.

Чтение локального корпуса в Jupyter без повторной загрузки из интернета:

```python
from datasets import load_dataset

corpus = load_dataset(
    "parquet",
    data_files="/workspace/data/opus_books/en-ru/*.parquet",
    split="train",
)
print(len(corpus))
print(corpus[0]["translation"]["ru"])
```

Русский текст каждой записи находится в `row["translation"]["ru"]`.


## Учебный notebook: первый шаг

Откройте `custom_russian_tokenizer.ipynb` в JupyterLab и выберите
**Run → Run All Cells**. Notebook читает локальный корпус и показывает:

- структуру записи и примеры русских текстов;
- пустые и отсутствующие строки, точные повторения;
- распределение длин в символах и словах;
- примеры служебных строк и возможных артефактов форматирования.

На этом шаге данные не очищаются и токенизатор не обучается.
Результаты диагностики станут основой правил очистки на следующем шаге.
