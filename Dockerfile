FROM python:3.11.5

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY main.py .

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "main.py", "--server.port", "8080"]