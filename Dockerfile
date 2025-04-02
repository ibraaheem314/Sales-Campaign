# Image de base
FROM python:3.9-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# Installation des dépendances système
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copie des requirements
COPY requirements.txt .

# Installation des packages Python
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install psycopg2-binary

# Copie du code
WORKDIR /app
COPY . .

# Ports exposés
EXPOSE 8501 5000

# Commande par défaut
CMD ["streamlit", "run", "dashboard/app.py"]