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

KEYCLOAK_SERVER_URL = "..."
KEYCLOAK_REALM = "..."
KEYCLOAK_CLIENT_ID = "..."
KEYCLOAK_CLIENT_SECRET = "..."

OID_TOKEN_URL = f'{KEYCLOAK_SERVER_URL}/auth/realms /{KEYCLOAK_REALM}/protocol/openid-connect/token'
PERMISSIONS_URL = f'{KEYCLOAK_SERVER_URL}/auth/realms /{KEYCLOAK_REALM}/authz/protection/permission'

WWW_FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"
JSON_CONTENT_TYPE = "application/json"

TICKET_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:uma-ticket"


def _encodeCacheData(res):
    return json.dumps({
        'accessToken': res.data.access_token,
        'refreshToken': res.data.refresh_token,
    }, separators=(',', ':'))


def _decodeCacheData(data):
    return json.loads(data)


# const _encodeCacheData = (res) = > {
#     return JSON.stringify({
#         accessToken: res.data.access_token,
#         refreshToken: res.data.refresh_token,
#     })
# }

# const _decodeCacheData = (data) = > {
#     return JSON.parse(data)
# }

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

# const _rolesToTags = (decodedToken) = > {
#     let tag = []
#     _.get(
#         decodedToken,
#         `resource_access.${process.env.KEYCLOAK_CLIENT_ID}.roles`,
#         []
#     ).forEach((role)=> {
#         if (TAG_ROLE_REGEXP.test(role)) {
#             tag.push(role.replace(TAG_ROLE_PREFIX, ""))
#         }
#     })
#     if (tag.length === 0) {
#         return ""
#     }
#     if (tag.length === 1) {
#         return ` ${tag[0]}`
#     }
#     throw new Error("User cannot have multiple tag roles assigned in Keycloak")
# }


cacheData = {}


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
    cacheData[username] = _encodeCacheData(x.text)
    print(cacheData)
    return f'allow{_rolesToTags(x.json())}'


# const authenticate = async (username, password) = > {
#     let call = {
#         method: "POST",
#         url: OID_TOKEN_URL,
#         headers: {
#             "Content-Type": WWW_FORM_CONTENT_TYPE,
#         },
#         data: Qs.stringify({
#             client_id: process.env.KEYCLOAK_CLIENT_ID,
#             client_secret: process.env.KEYCLOAK_CLIENT_SECRET,
#             username: username,
#             password: password,
#             grant_type: "password",
#             scope: "openid",
#         }),
#     }
#     let authResponse = await Axios(call)
#     let decodedToken = Jwt.decode(authResponse.data.access_token)
#     await Cache.set(username, _encodeCacheData(authResponse))
#     return `allow${_rolesToTags(decodedToken)}`
# }

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


# const _getUmaTicket = (accessToken, resource, scope) = > {
#     let call = {
#         method: "POST",
#         url: PERMISSIONS_URL,
#         headers: {
#             authorization: `Bearer ${accessToken}`,
#             "Content-Type": JSON_CONTENT_TYPE,
#         },
#         data: JSON.stringify([
#             {
#                 resource_id: resource,
#                 resource_scopes: [scope],
#             },
#         ]),
#     }
#     return Axios(call)
# }

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


# const _getRequestPartyTokenFromTicket = (accessToken, ticket) = > {
#     let call = {
#         method: "POST",
#         url: OID_TOKEN_URL,
#         headers: {
#             authorization: `Bearer ${accessToken}`,
#             "Content-Type": WWW_FORM_CONTENT_TYPE,
#         },
#         data: Qs.stringify({
#             client_id: process.env.KEYCLOAK_CLIENT_ID,
#             client_secret: process.env.KEYCLOAK_CLIENT_SECRET,
#             ticket: ticket,
#             grant_type: TICKET_GRANT_TYPE,
#         }),
#     }
#     return Axios(call)
# }

def authorize(username, resource, scope):
    authenticateTokens = _decodeCacheData(cacheData[username])
    # maybe error here
    umaTicket = _getUmaTicket(
        authenticateTokens.accessToken, resource, scope).data.ticket
    requestPartyToken = _getRequestPartyTokenFromTicket(
        authenticateTokens.accessToken,
        umaTicket
    ).access_token


# const authorize = async (username, resource, scope) = > {
#     let authenticateTokens = _decodeCacheData(await Cache.get(username))
#     let umaTicket = (
#         await _getUmaTicket(authenticateTokens.accessToken, resource, scope)
#     ).data.ticket
#     let requestPartyToken = (
#         await _getRequestPartyTokenFromTicket(
#             authenticateTokens.accessToken,
#             umaTicket
#         )
#     ).access_token
# }
