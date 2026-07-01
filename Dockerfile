FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/bootstrap_eval_set.py || true

ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000

EXPOSE 5000

CMD ["python", "web/app.py"]
