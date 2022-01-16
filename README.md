# Dissertation-HenriqueMatos

## Study of Docker
Lists all running and exited containers
> sudo docker ps -a
Lists all images
> sudo docker images
Lists all networks
> sudo docker network ls


Do a pull of every container i need

# Inspect container
inspect 
# Detached mode
-d

## Interactive mode 
use -it
## Port Mapping
-p 80:5000
## Volume mapping (Databases)
-v /opt/datadir:/var/lib/mysql

## Create environment variables
-e APP_COLOR=blue

## Open terminal
> sudo docker exec -it [NAME] /bin/bash
sudo docker exec -it teste_rabbitmq_1 /bin/bash


## Create a new image
> sudo docker build . -t henriquematos/imagetest

## Push image to DockerHub

> sudo docker push henriquematos/imagetest


## DNS server
Runs at address 127.0.0.11

## Store data in Volume
docker volume create data_volume
docker run -v data_volume:/var/lib/mysql mysql


## Run a Compose docker file
> sudo docker-compose up





## Run Keyloak

> sudo docker run -p 8080:8080 -e KEYCLOAK_USER=admin -e KEYCLOAK_PASSWORD=admin quay.io/keycloak/keycloak:16.1.0

