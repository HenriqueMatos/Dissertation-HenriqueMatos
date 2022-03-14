import json
from flask import Flask, request, render_template, redirect, flash, url_for
from flask import session, make_response, g
from flask import current_app
import requests
import time
import threading
import pika
import sys
import os
import redis
import pickle
# RegistrationForm,
from forms import  LoginForm
from keycloak_utils import get_oidc, get_token, check_token


class EdgeData():
    def __init__(self, sensorID, authorization):
        self.sensorID = sensorID
        self.authorization = authorization


r = redis.Redis(host="localhost", port="6379")

app = Flask(__name__)
app.config.from_object('settings')


@app.before_request
def load_user():
    # print("AQUI")
    g.username = session.get('username')
    g.access_token = session.get('access_token')
    print(g.username, g.access_token)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():
        oidc_obj = get_oidc()
        print(form.username.data, form.password.data)
        token = get_token(oidc_obj, form.username.data, form.password.data)
        print("\nTOKEN: %s\n" % token)
        response = make_response(redirect(url_for('home')))
        if token:
            response.set_cookie('access_token', token['access_token'])
            session['access_token'] = token['access_token']
            session['username'] = form.username.data
        return response
    return render_template('login.html', form=form)


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('access_token', None)
    return redirect(url_for('home'))


@app.route('/headers')
def headers():
    return dict(request.headers)


@app.route('/protected')
def protected():
    ingress_host = current_app.config.get('INGRESS_HOST')
    resp = 'Forbidden!'
    access_token = session.get('access_token')
    if access_token:
        if check_token(access_token):
            headers = {'Authorization': 'Bearer ' + access_token}
            r = requests.get(ingress_host, headers=headers)
            resp = 'Protected resource is accessible. Yay! Here is the response: %s' % str(
                r.text)
    return resp


def web():
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


def runningWorker(DataServer):
    credentials = pika.PlainCredentials('admin', 'admin')

    connection_parameters = pika.ConnectionParameters(
        'localhost', credentials=credentials, virtual_host="keycloak_test")
    connection = pika.BlockingConnection(
        connection_parameters)
    channel = connection.channel()

    channel.queue_declare(queue='hello')

    def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)

    channel.basic_consume(
        queue='hello', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':

    if r.exists("data"):
        DataServer = json.loads(r.get("data"))
    else:
        DataServer = []

    threading.Thread(target=web, daemon=False).start()
    threading.Thread(target=runningWorker, daemon=False,
                     args=DataServer).start()
    while True:
        time.sleep(1)
    #     for thread in threading.enumerate():
    #         print(thread.name)
