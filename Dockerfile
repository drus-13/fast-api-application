FROM python:3.9
WORKDIR /app
COPY . /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements/requirements.txt
EXPOSE 8501
ENTRYPOINT ["bash", "run.sh"] # For deployment to Heroku
CMD ["app/main.py"]