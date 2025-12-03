# Imagen base de Python
FROM python:3.9-slim

# Evitar buffer en logs
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto (app, data, models, etc.)
COPY . .

# Puerto donde corre Dash
EXPOSE 8050

# Comando para ejecutar la app
CMD ["python", "app.py"]
