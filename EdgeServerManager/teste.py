from keycloak import KeycloakOpenID

# Configure client
keycloak_openid = KeycloakOpenID(server_url="http://localhost:8080/auth/",
                    client_id="EdgeServer1",
                    realm_name="AppAuthenticator",
                    client_secret_key="9Wokn60OmIU13WofzyrZzDeKhyIkkvRk")

# Get WellKnow
config_well_know = keycloak_openid.well_know()

# Get Token
token = keycloak_openid.token("edgeserver", "edgeserver")

print(token)