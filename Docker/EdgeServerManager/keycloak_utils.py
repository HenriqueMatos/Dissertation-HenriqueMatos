from keycloak import KeycloakAdmin, KeycloakOpenID
from flask import current_app


def verifyToken(oidc_obj, Authentication):
    try:
        userinfo = oidc_obj.userinfo(Authentication)
    except:
        return False,False
    # print(userinfo)
    return True,userinfo

    return userinfo


def get_oidc2(SERVER_URL, CLIENT_ID, REALM_NAME, CLIENT_SECRET):
    keycloak_openid = KeycloakOpenID(server_url=SERVER_URL,
                                     client_id=CLIENT_ID,
                                     realm_name=REALM_NAME,
                                     client_secret_key=CLIENT_SECRET)
    # print(keycloak_openid.well_know())

    return keycloak_openid


def get_oidc():
    keycloak_openid = KeycloakOpenID(server_url=current_app.config.get('SERVER_URL'),
                                     client_id=current_app.config.get(
                                         'CLIENT_ID'),
                                     realm_name=current_app.config.get(
                                         'REALM_NAME'),
                                     client_secret_key=current_app.config.get(
        'CLIENT_SECRET'))
    # print(keycloak_openid.well_know())
    return keycloak_openid


def get_token(oidc_obj, username, password):
    try:
        return oidc_obj.token(username, password)
    except Exception as e:
        print("Exception occurs: %s" % e)
    return None


def check_token(access_token):
    oidc = get_oidc()
    token_info = oidc.introspect(access_token)
    if token_info.get('active'):
        return True
    return False


def get_userinfo(access_token):
    oidc = get_oidc()
    try:
        return oidc.userinfo(access_token)
    except Exception as e:
        print("Exception occurs: %s" % e)
    return None
