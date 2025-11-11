FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
