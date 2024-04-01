FROM python:3.11.5

WORKDIR /app

RUN pip install --upgrade pip

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]