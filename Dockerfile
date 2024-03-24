FROM python:3.11

WORKDIR /API_Interface
COPY main.py .
COPY Neurall.py .
COPY requirements.txt /
COPY .env /API_Interface/.env
RUN pip install -r /requirements.txt

EXPOSE 5000

CMD ["python", "main.py"]