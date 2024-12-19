#Koristimo zvaničnu Python bazu
FROM python:3.9-slim

#Postavimo radni direktorijum u kontejneru
WORKDIR /app

#Kopiramo sve fajlove iz lokalnog foldera u kontejner
COPY . .

#Instaliramo sve potrebne biblioteke (navedene u requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

#Ekspoziramo port za pokretanje aplikacije
EXPOSE 5000

#Komanda za pokretanje aplikacije
CMD ["python", "main.py"]