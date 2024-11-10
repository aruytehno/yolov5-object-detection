# Dockerfile
FROM python:3.10-slim

# Установим необходимые библиотеки для работы графического интерфейса
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip до последней версии
RUN pip install --upgrade pip

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt /app/requirements.txt

# Устанавливаем зависимости
# Устанавливаем sahi до установки opencv-python, чтобы избежать конфликта версий
RUN pip install --no-cache-dir "sahi<=0.11.18" && \
    pip install --no-cache-dir -r /app/requirements.txt

# Копируем исходный код проекта в контейнер
COPY . /app

# Запуск приложения
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
