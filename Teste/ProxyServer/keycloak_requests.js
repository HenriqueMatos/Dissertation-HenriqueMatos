import Axios from "axios";
import Qs from "qs";
import Jwt from "jsonwebtoken";
import _ from "lodash";
import Cache from "./cache";

const SCOPES = {
  vhost: {
    access: "vhost_access",
  },
  resource: {
    read: "read",
    write: "write",
    configure: "configure",
  },
  topic: {
    read: "read",
    write: "write",
  },
};

const TAG_ROLE_PREFIX = "rabbitmq-tag-";
const TAG_ROLE_REGEXP = RegExp(`${TAG_ROLE_PREFIX}\\w+`);

const OID_TOKEN_URL = `${process.env.KEYCLOAK_SERVER_URL}/auth/realms/${process.env.KEYCLOAK_REALM}/protocol/openid-connect/token`;
const PERMISSIONS_URL = `${process.env.KEYCLOAK_SERVER_URL}/auth/realms/${process.env.KEYCLOAK_REALM}/authz/protection/permission`;

const WWW_FORM_CONTENT_TYPE = "application/x-www-form-urlencoded";
const JSON_CONTENT_TYPE = "application/json";

const TICKET_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:uma-ticket";

const _encodeCacheData = (res) => {
  return JSON.stringify({
    accessToken: res.data.access_token,
    refreshToken: res.data.refresh_token,
  });
};

const _decodeCacheData = (data) => {
  return JSON.parse(data);
};

const _rolesToTags = (decodedToken) => {
  let tag = [];
  _.get(
    decodedToken,
    `resource_access.${process.env.KEYCLOAK_CLIENT_ID}.roles`,
    []
  ).forEach((role) => {
    if (TAG_ROLE_REGEXP.test(role)) {
      tag.push(role.replace(TAG_ROLE_PREFIX, ""));
    }
  });
  if (tag.length === 0) {
    return "";
  }
  if (tag.length === 1) {
    return ` ${tag[0]}`;
  }
  throw new Error("User cannot have multiple tag roles assigned in Keycloak");
};

const authenticate = async (username, password) => {
  let call = {
    method: "POST",
    url: OID_TOKEN_URL,
    headers: {
      "Content-Type": WWW_FORM_CONTENT_TYPE,
    },
    data: Qs.stringify({
      client_id: process.env.KEYCLOAK_CLIENT_ID,
      client_secret: process.env.KEYCLOAK_CLIENT_SECRET,
      username: username,
      password: password,
      grant_type: "password",
      scope: "openid",
    }),
  };
  let authResponse = await Axios(call);
  let decodedToken = Jwt.decode(authResponse.data.access_token);
  await Cache.set(username, _encodeCacheData(authResponse));
  return `allow${_rolesToTags(decodedToken)}`;
};

const _getUmaTicket = (accessToken, resource, scope) => {
  let call = {
    method: "POST",
    url: PERMISSIONS_URL,
    headers: {
      authorization: `Bearer ${accessToken}`,
      "Content-Type": JSON_CONTENT_TYPE,
    },
    data: JSON.stringify([
      {
        resource_id: resource,
        resource_scopes: [scope],
      },
    ]),
  };
  return Axios(call);
};

const _getRequestPartyTokenFromTicket = (accessToken, ticket) => {
  let call = {
    method: "POST",
    url: OID_TOKEN_URL,
    headers: {
      authorization: `Bearer ${accessToken}`,
      "Content-Type": WWW_FORM_CONTENT_TYPE,
    },
    data: Qs.stringify({
      client_id: process.env.KEYCLOAK_CLIENT_ID,
      client_secret: process.env.KEYCLOAK_CLIENT_SECRET,
      ticket: ticket,
      grant_type: TICKET_GRANT_TYPE,
    }),
  };
  return Axios(call);
};

const authorize = async (username, resource, scope) => {
  let authenticateTokens = _decodeCacheData(await Cache.get(username));
  let umaTicket = (
    await _getUmaTicket(authenticateTokens.accessToken, resource, scope)
  ).data.ticket;
  let requestPartyToken = (
    await _getRequestPartyTokenFromTicket(
      authenticateTokens.accessToken,
      umaTicket
    )
  ).access_token;
};

module.exports = { authenticate, authorize, SCOPES };
