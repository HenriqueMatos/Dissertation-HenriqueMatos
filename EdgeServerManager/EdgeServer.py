import codecs
import json
import re
from traceback import print_tb
from flask import Flask, request, render_template, redirect, flash, url_for
from flask import session, make_response, g, jsonify
from flask import current_app
import numpy as np
from PIL import Image
import base64
import io
import requests
import time
import threading
import pika
import sys
import os
import redis
import pickle
# RegistrationForm,
from forms import LoginForm
from keycloak_utils import get_oidc2, get_oidc, get_token, check_token, verifyToken


SERVER_URL = "http://localhost:8080/auth/"
REALM_NAME = "AppAuthenticator"
CLIENT_ID = "EdgeServer1"
CLIENT_SECRET = "deCGEfmNbxFkC5z32UnwxtyQThTx4Evy"

r = redis.Redis(host="localhost", port="6379")

global DataServer

if r.exists("data"):
    DataServer = json.loads(r.get("data"))
else:
    DataServer = []

app = Flask(__name__)
app.config.from_object('settings')

oidc_obj = get_oidc2(SERVER_URL, CLIENT_ID, REALM_NAME, CLIENT_SECRET)


@app.before_request
def load_user():
    # print("AQUI")
    g.dataserver = DataServer
    g.username = session.get('username')
    g.access_token = session.get('access_token')
    # print(g.username, g.access_token)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/updateDataServer', methods=['GET'])
def updateDataServer():
    return jsonify(DataServer=DataServer)


@app.route('/config', methods=['POST'])
def configObject():
    config = {}
    if request.method == "POST":
        preferred_username = request.form["preferred_username"]
        data = {}
        data["preferred_username"] = preferred_username
        for x in DataServer:
            if x["preferred_username"] == preferred_username:
                configData = json.loads(x["config"])
                data["config"] = configData
                data["sensorid"] = x["camera_id"]
                imageData = x["frame"]
                print(x["config"])
                for item in configData:
                    if type(configData[item]) is dict:
                        config[item] = list(configData[item].keys())
                    elif type(configData[item]) is list:
                        config[item] = "list"
                    else:
                        config[item] = None
                break

    return render_template('config.html', config=config, data=data, image=base64.b64encode(base64.decodebytes(imageData.encode('utf-8'))).decode('utf-8'))


KEYS_line_intersection_zone = {
    'name': 'Teste 123',
    'start_point':  ([],"list"),
    'end_point':  ([],"list"),
    'zone_direction_1or2': 1,
    'name_zone_before': 'outside',
    'name_zone_after': 'inside'
}
KEYS_zone = {
    'name_inside_zone': 'pátio escola de engenharia',
    'name_outside_zone': 'pátio escola de engenharia',
    'points': ([],"listoflist")
}
KEYS_remove_area = [([],"listoflist")]


@app.route('/config_points', methods=['POST'])
def configPoints():
    keys = None
    if request.method == "POST":
        print(request.form)
        preferred_username = request.form["preferred_username"]
        configString = request.form["config"]
        configString1 = request.form["config1"]
        for x in DataServer:
            if x["preferred_username"] == preferred_username:
                imageData = x["frame"]
                Config = json.loads(x["config"])
                data = Config[configString][configString1]
                data2 = []
                if configString1 == "line_intersection_zone":
                    keys = KEYS_line_intersection_zone
                elif configString1 == "zone":
                    keys = KEYS_zone
                elif configString1 == "remove_area":
                    keys = KEYS_remove_area
                # for each_data in data:
                #     if type(each_data) != list:
                #         aux = {}
                #         for item in each_data:
                #             if type(each_data[item]) == list:
                #                 if type(each_data[item][0]) == list:
                #                     aux[item] = (each_data[item], "listoflist")
                #                 else:   
                #                     aux[item] = (each_data[item], "list")
                #             else:
                #                 aux[item] = (each_data[item], "not")
                #         data2.append(aux)
                #     else:
                #         data2.append(each_data)
                
                break

    return render_template('config_points.html', keys=keys, config=configString1, data=data, image=base64.b64encode(base64.decodebytes(imageData.encode('utf-8'))).decode('utf-8'))


def web():
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


def runningWorker():
    credentials = pika.PlainCredentials('admin', 'admin')

    connection_parameters = pika.ConnectionParameters(
        'localhost', credentials=credentials, virtual_host="keycloak_test")
    connection = pika.BlockingConnection(
        connection_parameters)
    channel = connection.channel()

    channel.queue_declare(queue='hello')

    def callback(ch, method, properties, body):
        # print(" [x] Received %r" % body)

        receivedObject = json.loads(body)
        if receivedObject.__contains__("type"):
            if receivedObject["type"] == "login":
                check, data = verifyToken(
                    oidc_obj, receivedObject["Authenticate"])
                # print(data)
                if check:
                    flag = True
                    Findindex = -1
                    for i, d in enumerate(DataServer):
                        if d['preferred_username'] == data["preferred_username"]:
                            flag = False
                            Findindex = i
                            break
                    else:
                        i = -1
                    if flag is False:
                        DataServer.pop(Findindex)
                    receivedObject["preferred_username"] = data["preferred_username"]
                    DataServer.append(receivedObject)
    channel.basic_consume(
        queue='hello', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':

    threading.Thread(target=web, daemon=False).start()
    threading.Thread(target=runningWorker, daemon=False).start()
    while True:
        time.sleep(5)
        # g.dataserver = DataServer
        # print(DataServer)
    #     for thread in threading.enumerate():
    #         print(thread.name)
