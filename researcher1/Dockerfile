FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

VOLUME ["/artifacts"]

CMD ["python", "train.py"]
