# export FLASK_ENV=development
# export FLASK_APP=main
# flask run -h localhost -p 3000
from os import access
from flask import Flask, redirect, url_for, request
import requests
import re
import json


SCOPES = {
    'vhost': {
        'access': "vhost_access",
    },
    'resource': {
        'read': "read",
        'write': "write",
        'configure': "configure",
    },
    'topic': {
        'read': "read",
        'write': "write",
    },
}

TAG_ROLE_PREFIX = "rabbitmq-tag-"
TAG_ROLE_REGEXP = "/rabbitmq-tag-\w +/"

KEYCLOAK_SERVER_URL = "http://localhost:8080"
KEYCLOAK_REALM = "myrealm"
KEYCLOAK_CLIENT_ID = "Rabbitmq-oauth2-proxyapp"
KEYCLOAK_CLIENT_SECRET = "dU1dOCjyo9jGKP4DoDR6Gc7TIqIgVtPz"

OID_TOKEN_URL = f'{KEYCLOAK_SERVER_URL}/auth/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token'
PERMISSIONS_URL = f'{KEYCLOAK_SERVER_URL}/auth/realms/{KEYCLOAK_REALM}/authz/protection/permission'

WWW_FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"
JSON_CONTENT_TYPE = "application/json"

TICKET_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:uma-ticket"


cacheData = {}


def _encodeCacheData(res):
    return json.dumps({
        'accessToken': res.data.access_token,
        'refreshToken': res.data.refresh_token,
    }, separators=(',', ':'))


def _decodeCacheData(data):
    return json.loads(data)


def _rolesToTags(decodedToken):
    tag = []
    print(decodedToken)
    for item in decodedToken:
        if re.search(TAG_ROLE_PREFIX, item['resource_access'][KEYCLOAK_CLIENT_ID]['roles']):
            tag.insert(item['resource_access'][KEYCLOAK_CLIENT_ID]
                       ['roles'].replace(TAG_ROLE_PREFIX, ""))
    if len(tag) == 0:
        return ""
    else:
        return f' {tag[0]}'


def authenticate(username, password):
    myobj = {
        "client_id": KEYCLOAK_CLIENT_ID,
        "client_secret": KEYCLOAK_CLIENT_SECRET,
        "username": username,
        "password": password,
        "grant_type": "password",
        "scope": "openid",
    }

    x = requests.post(OID_TOKEN_URL, data=myobj, headers={
                      "Content-Type": WWW_FORM_CONTENT_TYPE})

    print(x.json())
    # cacheData.append(username)
   #  cacheData[username] = _encodeCacheData(x.text)
    cacheData[username] = x.json()
    print(cacheData)
    return f'allow{_rolesToTags(x.json())}'


def _getUmaTicket(accessToken, resource, scope):
    myobj = [
        {
            "resource_id": resource,
            "resource_scopes": [scope],
        },
    ]

    x = requests.post(PERMISSIONS_URL, data=myobj, headers={
        "authorization": f'Bearer {accessToken}',
        "Content-Type": JSON_CONTENT_TYPE,
    })
    return(x.text)


def _getRequestPartyTokenFromTicket(accessToken, ticket):
    myobj = {
        "client_id": KEYCLOAK_CLIENT_ID,
        "client_secret": KEYCLOAK_CLIENT_SECRET,
        "ticket": ticket,
        "grant_type": TICKET_GRANT_TYPE,
    }

    x = requests.post(OID_TOKEN_URL, data=myobj, headers={
        "authorization": f'Bearer {accessToken}',
        "Content-Type": WWW_FORM_CONTENT_TYPE,
    })
    return(x.text)


def authorize(username, resource, scope):
    authenticateTokens = _decodeCacheData(cacheData[username])
    # maybe error here
    umaTicket = _getUmaTicket(
        authenticateTokens.accessToken, resource, scope).data.ticket
    requestPartyToken = _getRequestPartyTokenFromTicket(
        authenticateTokens.accessToken,
        umaTicket
    ).access_token
    print(authenticateTokens)
    print(umaTicket)
    print(requestPartyToken)


app = Flask(__name__)


@app.route('/')
def main():
    return "Hello Guys"


@app.route('/user', methods=['POST'])
def user():
    print(request.form['username'])
    if isinstance(request.form['username'], str) and isinstance(request.form['password'], str):

        authResponse = authenticate(
            request.form['username'], request.form['password'])
        return authResponse
    else:
        return 'deny'


@app.route('/vhost', methods=['POST'])
def vhost():
    if isinstance(request.form['username'], str) and isinstance(request.form['vhost'], str) and isinstance(request.form['ip'], str) and isinstance(request.form['tags'], str):

        authorize(request.form['username'],
                  request.form['vhost'], SCOPES['vhost']['access'])
        return 'allow'
    else:
        return 'deny'


@app.route('/resource', methods=['POST'])
def resource():
    if isinstance(request.form['username'], str) and isinstance(request.form['vhost'], str) and isinstance(request.form['resource'], str) and isinstance(request.form['name'], str) and isinstance(request.form['permission'], str) and isinstance(request.form['tags'], str):

        authorize(request.form['username'],
                  request.form['name'], request.form['permission'])
        return 'allow'
    else:
        return 'deny'


@app.route('/topic', methods=['POST'])
def topic():
    if isinstance(request.form['username'], str) and isinstance(request.form['vhost'], str) and isinstance(request.form['resource'], str) and isinstance(request.form['name'], str) and isinstance(request.form['permission'], str) and isinstance(request.form['routing_key'], str) and isinstance(request.form['tags'], str):

        authorize(request.form['username'],
                  request.form['name'], permissionName(request.form['permission'], request.form['routing_key']))
        return 'allow'
    else:
        return 'deny'


def permissionName(permission, routing_key):
    return f'{permission}_with_rk_{routing_key}'


if __name__ == '__main__':
    app.run(debug=True)


# @app.route('/vhost', methods=['POST'])
# def vhost(args):

# @app.route('/resource', methods=['POST'])
# def resource(args):

# @app.route('/topic', methods=['POST'])
# def topic(args):
