import codecs
import copy
from enum import Flag
import json
import re
from datetime import datetime
import paho.mqtt.client as mqtt
from traceback import print_tb
from flask import Flask, request, render_template, redirect, flash, url_for
from flask import session, make_response, g, jsonify
from flask import current_app
from matplotlib.font_manager import json_load
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
# from forms import LoginForm
from keycloak_utils import get_oidc2, get_oidc, get_token, check_token, verifyToken
# from torchreid.models.nasnet import BranchSeparables

SERVER_IP = "192.168.233.139"

SERVER_URL = "http://"+SERVER_IP+":8080/auth/"
REALM_NAME = "AppAuthenticator"
CLIENT_ID = "EdgeServer1"
CLIENT_SECRET = "deCGEfmNbxFkC5z32UnwxtyQThTx4Evy"

r = redis.Redis(host=SERVER_IP, port="6379")

global client
global DataServer

if r.exists("data"):
    # DataServer = r.hget("data")
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
    config = {
        "ip": "noneditable",
        "camera_id": "noneditable",
        "camera_name": "editable",
        "camera_zone": "editable",
        "timestamp_config_creation": "noneditable",
        "weights": "noneditable",
        "source": "noneditable",
        "iou_thres": "editable",
        "conf_thres": "editable",
        "img_size": "noneditable",
        "cmc_method": "noneditable",
        "track_high_thresh": "editable",
        "track_low_thresh": "editable",
        "new_track_thresh": "editable",
        "aspect_ratio_thresh": "editable",
        "classes": "editable",
        "folder_remove_seconds": "editable",
        "ReID_mean_threshold": "editable",
        "ReID_median_threshold": "editable",
        "ReID_mode_threshold": "editable",
        "cam_coordinates": "noneditable",
        "global_map_scale": "editable",
        "global_map_angle": "editable",
        "global_map_offset": "editable",
        "input": "button"
    }
    if request.method == "POST":
        preferred_username = request.form["preferred_username"]
        data = {}
        data["preferred_username"] = preferred_username
        for x in DataServer:
            if x["preferred_username"] == preferred_username:
                configData = x["config"]
                data["config"] = configData
                data["sensorid"] = x["camera_id"]
                print(x["config"])
                break
    return render_template('config.html', config=config, data=data)


KEYS_line_intersection_zone = {
    'name': 'Teste 123',
    'start_point':  ([], "list"),
    'end_point':  ([], "list"),
    'name_zone_before': 'outside',
    'name_zone_after': 'inside',
    'id_association': {}
}
KEYS_zone = {
    'name_inside_zone': 'pátio escola de engenharia',
    'name_outside_zone': 'pátio escola de engenharia',
    'points': ([], "listoflist")
}
KEYS_remove_area = [([], "listoflist")]


@app.route('/refresh_all', methods=['GET'])
def refresh_all():
    sendData = {
        "type": "refresh"
    }
    for index in range(len(DataServer)):
        DataServer[index]["status"] = False
        client.publish("edge_config/"+DataServer[index]["preferred_username"],
                       json.dumps(sendData))
    r.set("data", json.dumps(DataServer))
    return "OK"


@app.route('/config_points', methods=['POST'])
def configPoints():
    keys = None
    if request.method == "POST":
        print(request.form)
        preferred_username = request.form["preferred_username"]
        config_name_before = request.form["config"]
        config_name = request.form["config1"]

        for index in range(len(DataServer)):
            if DataServer[index]["preferred_username"] == preferred_username:
                imageData = DataServer[index]["frame"]
                data=copy.deepcopy(DataServer[index]["config"][config_name_before][config_name])

                if config_name == "line_intersection_zone":
                    for index2 in range(len(data)):
                        data[index2].pop('id_association')
                    keys = KEYS_line_intersection_zone
                elif config_name == "zone":
                    keys = KEYS_zone
                elif config_name == "remove_area":
                    keys = KEYS_remove_area
                break


    return render_template('config_points.html', keys=keys, preferred_username=preferred_username, configBefore=config_name_before, config=config_name, data=data, image=base64.b64encode(base64.decodebytes(imageData.encode('utf-8'))).decode('utf-8'))


@app.route('/config_points_set', methods=['POST'])
def SetconfigPoints():
    if request.method == "POST":
        data = json.loads(request.form["data"])
        preferred_username = data["preferred_username"]
        config_name_before = data["configBefore"]
        config_name = data["config"]
        new_data = data["data"]
        print("CONA",preferred_username, config_name_before, config_name)
        print(new_data)

        for index in range(len(DataServer)):
            if DataServer[index]["preferred_username"] == preferred_username:
                # print(DataServer[index]["config"]
                #       [config_name_before][config_name])
                # SAVE IN DATABASE (REDIS)
                if(DataServer[index]["config"][config_name_before][config_name] != new_data):
                    # Send to Camera
                    # print(DataServer[index]["config"][config_name_before][config_name])
                    # print("NEW")
        
                    # if(config_name=="line_intersection_zone"):
                    #     for index, value in enumerate(new_data):
                    #         if len(DataServer[index]["config"][config_name_before][config_name])<=index:
                    #             DataServer[index]["config"][config_name_before][config_name].append(value)
                    #         else:
                    #             for key,name in value.items():
                    #                 DataServer[index]["config"][config_name_before][config_name][index][key]=name
                            
                    # else:
                    #     DataServer[index]["config"][config_name_before][config_name] = new_data
                    # # Config[config_name_before][config_name]=new_data

                    for index2, value in enumerate(new_data):
                        if len(DataServer[index]["config"][config_name_before][config_name])<=index2:
                            DataServer[index]["config"][config_name_before][config_name].append(value)
                        else:
                            DataServer[index]["config"][config_name_before][config_name][index2].update(value)
                    print(DataServer[index]["config"][config_name_before][config_name])
                else:
                    print("OLD")
                break

    return "OK"


@app.route('/config_set', methods=['POST'])
def Setconfig():
    if request.method == "POST":
        # print(request.form["data"])
        data = json.loads(request.form["data"])
        preferred_username = data["preferred_username"]
        new_data = data["data"]
        print("AQUI")
        print(preferred_username)
        print(new_data)

        for index in range(len(DataServer)):
            if DataServer[index]["preferred_username"] == preferred_username:
                print(DataServer[index]["config"])
                for item in new_data:
                    DataServer[index]["config"][item] = new_data[item]
                # SAVE IN DATABASE (REDIS)
                # SEND NEW CONFIG TO CAMERA
                sendData = {}
                sendData["type"] = "update"
                sendData["config"] = DataServer[index]["config"]
                client.publish("edge_config/"+preferred_username,
                               json.dumps(sendData))

        return "OK"


@app.route('/id_association', methods=['GET'])
def id_association():
    ID_associations = {}
    ID_associationTuple = []
    DataObject = []

    for value in DataServer:
        NoAssociationIndexes = []
        for index, eachLine in enumerate(value["config"]["input"]["line_intersection_zone"]):
            # print(eachLine)
            if eachLine.__contains__('id_association'):
                if eachLine['id_association']:
                    # HAS ASSOCIATION
                    ID_associations[eachLine['name']
                                    ] = {"name": eachLine['id_association']['name'],
                                         "preferred_username": value["preferred_username"]}
                else:
                    NoAssociationIndexes.append(index)
            else:
                NoAssociationIndexes.append(index)

        if len(NoAssociationIndexes) > 0:
            DataObject.append({
                "frame":  base64.b64encode(base64.decodebytes(value["frame"].encode('utf-8'))).decode('utf-8'),
                "preferred_username": value["preferred_username"],
                "line_intersection_zone": [value["config"]["input"]["line_intersection_zone"][x] for x in NoAssociationIndexes]
            })
    Associations = []
    for key, value in ID_associations.items():
        if key in [value2["name"] for value2 in ID_associations.values()]:
            notisInTuple = False

            for eachTuple in Associations:
                if key in eachTuple:
                    notisInTuple = True
                    break

            if notisInTuple == False:
                print([key, value["preferred_username"], value["name"],
                                     ID_associations[value["name"]]["preferred_username"]])
                Associations.append([key, value["preferred_username"], value["name"],
                                     ID_associations[value["name"]]["preferred_username"]])

    print("FFFFFFFFFF", Associations)

    return render_template('id_association.html', DataObject=DataObject, Associations=Associations)


@app.route('/id_association/get_association', methods=['GET'])
def getAssociations():
    ID_associations = {}

    for value in DataServer:
        for index, eachLine in enumerate(value["config"]["input"]["line_intersection_zone"]):
            # print(eachLine)
            if eachLine.__contains__('id_association'):
                if eachLine['id_association']:
                    # HAS ASSOCIATION
                    ID_associations[eachLine['name']
                                    ] = {"name": eachLine['id_association']['name'],
                                         "preferred_username": value["preferred_username"]}
    Associations = []
    for key, value in ID_associations.items():
        if key in [value2["name"] for value2 in ID_associations.values()]:
            notisInTuple = False

            for eachTuple in Associations:
                if key in eachTuple:
                    notisInTuple = True
                    break

            if notisInTuple == False:
                Associations.append([key, value["preferred_username"], value["name"],
                                     ID_associations[value["name"]]["preferred_username"]])

    return jsonify(Associations=Associations)


@app.route('/id_association/remove_association', methods=['POST'])
def RemoveAssociation():
    if request.method == "POST":
        print(request.form)
        data = json.loads(request.form["data"])
        # data = json.loads(request.form["data"])
        print(data)
        for eachData in data:
            print(eachData["Name"])
            print(eachData["ConfigName"])
        Names = [value2["Name"] for value2 in data]
        ConfigNames = [value2["ConfigName"] for value2 in data]

        for index in range(len(DataServer)):
            if DataServer[index]["preferred_username"] in Names:
                indexToRemove = Names.index(
                    DataServer[index]["preferred_username"])

                for index_line_intersection_Config in range(len(DataServer[index]["config"]["input"]["line_intersection_zone"])):
                    if DataServer[index]["config"]["input"]["line_intersection_zone"][index_line_intersection_Config]["name"] == ConfigNames[indexToRemove]:
                        DataServer[index]["config"]["input"]["line_intersection_zone"][index_line_intersection_Config]["id_association"] = {
                        }
                        # SEND TO TRACKING SYSTEMS
                        sendData = {}
                        sendData["type"] = "update"
                        sendData["config"] = DataServer[index]["config"]
                        client.publish("edge_config/"+DataServer[index]["preferred_username"],
                                       json.dumps(sendData))

                Names.pop(indexToRemove)
                ConfigNames.pop(indexToRemove)

        return "OK"


@app.route('/id_association/add_association', methods=['POST'])
def AddAssociation():
    if request.method == "POST":
        print(request.form)
        data = json.loads(request.form["data"])
        # data = json.loads(request.form["data"])
        print(data)
        for eachData in data:
            print(eachData["Name"])
            print(eachData["ConfigName"])
        Names = [value2["Name"] for value2 in data]
        ConfigNames = [value2["ConfigName"] for value2 in data]

        for index in range(len(DataServer)):
            if DataServer[index]["preferred_username"] in Names:
                indexToAdd = Names.index(
                    DataServer[index]["preferred_username"])

                for index_line_intersection_Config in range(len(DataServer[index]["config"]["input"]["line_intersection_zone"])):
                    if DataServer[index]["config"]["input"]["line_intersection_zone"][index_line_intersection_Config]["name"] == ConfigNames[indexToAdd]:
                        if(indexToAdd == 0):
                            DataServer[index]["config"]["input"]["line_intersection_zone"][index_line_intersection_Config]["id_association"] = {
                                "publish_location": "edge_config/"+Names[1],
                                "name": ConfigNames[1]
                            }
                        if(indexToAdd == 1):
                            DataServer[index]["config"]["input"]["line_intersection_zone"][index_line_intersection_Config]["id_association"] = {
                                "publish_location": "edge_config/"+Names[0],
                                "name": ConfigNames[0]
                            }
                        # SEND TO TRACKING SYSTEMS
                        sendData = {}
                        sendData["type"] = "update"
                        sendData["config"] = DataServer[index]["config"]
                        client.publish("edge_config/"+DataServer[index]["preferred_username"],
                                       json.dumps(sendData))

                # Names.pop(indexToAdd)
                # ConfigNames.pop(indexToAdd)

        return "OK"


def web():
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)


# def runningWorker():
def on_message(client, userdata, message):
        print("Received message: ", str(message.payload.decode("utf-8")),
              " From: ", message.topic, " ")
        receivedObject = json.loads(message.payload)
        if receivedObject.__contains__("type"):
            if receivedObject["type"] == "login":
                check, data = verifyToken(
                    oidc_obj, receivedObject["Authenticate"])
                print(check, data)
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
                    for key in receivedObject:
                        # print(key)
                        if key == "config":
                            # print(receivedObject[key])
                            receivedObject[key] = json.loads(
                                receivedObject[key])
                    now = datetime.now()
                    receivedObject["refresh_timestamp"] = datetime.timestamp(
                        now)
                    receivedObject["refresh_date"] = now.strftime(
                        "%d/%m/%Y %H:%M:%S")
                    receivedObject["status"] = True
                    DataServer.append(receivedObject)
                r.set("data", json.dumps(DataServer))



if __name__ == '__main__':
    print("START")
    
    client = mqtt.Client("EdgeServer1")

    threading.Thread(target=web, daemon=False).start()
    
    client.connect(SERVER_IP)
    client.subscribe("camera_config")
    client.on_message = on_message
    client.loop_start()
    while(1):
        print("AQUI")        
        time.sleep(2)



