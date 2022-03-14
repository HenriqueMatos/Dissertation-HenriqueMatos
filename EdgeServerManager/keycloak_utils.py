from keycloak import KeycloakAdmin, KeycloakOpenID
from flask import current_app



def get_oidc():
    keycloak_openid = KeycloakOpenID(server_url=current_app.config.get('SERVER_URL'),
                                     client_id=current_app.config.get(
                                         'CLIENT_ID'),
                                     realm_name=current_app.config.get(
                                         'REALM_NAME'),
                                     client_secret_key=current_app.config.get(
        'CLIENT_SECRET'))
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
