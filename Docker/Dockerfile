FROM ubuntu:latest

RUN apt update 
RUN apt install python3 -y
RUN apt install python3-pip -y

RUN pip3 install python-dotenv
RUN pip3 install paho-mqtt
RUN pip3 install flask
RUN pip3 install matplotlib
RUN pip3 install requests
RUN pip3 install pika
RUN pip3 install redis
RUN pip3 install python-keycloak

WORKDIR /usr/app/src
COPY ./EdgeServerManager ./
EXPOSE 5000
CMD ["python3", "EdgeServer.py"]
