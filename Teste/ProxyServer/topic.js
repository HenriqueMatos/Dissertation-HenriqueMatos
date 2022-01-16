import Joi              from '@hapi/joi';
import KeycloakRequests from './keycloak_requests';

const validate = async (obj)=> {
  let schema = Joi.object({
    username: Joi.string().required(),
    vhost: Joi.string().required(),
    resource: Joi.string().required().allow('topic'),
    name: Joi.string().required(),
    permission: Joi.string().required().allow(...Object.values(KeycloakRequests.SCOPES.topic)),
    routing_key: Joi.string().required(),
    tags: Joi.string().required(),
    'variable_map.username': Joi.string(),
    'variable_map.vhost': Joi.string()
  });
  await schema.validateAsync(obj);
};

const permissionName = (req)=> {
  return `${req.body.permission}_with_rk_${req.body.routing_key}`;
};

const TopicEndpoint = async (req, res)=> {
  try {
    await validate(req.body);
    await KeycloakRequests.authorize(req.body.username, req.body.name, permissionName(req));
    res.send('allow');    
  } catch (e) {
    res.send('deny');
  }
};
export default TopicEndpoint;