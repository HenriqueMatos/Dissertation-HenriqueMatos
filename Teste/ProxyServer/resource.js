import Joi              from '@hapi/joi';
import KeycloakRequests from './keycloak_requests';

const validate = async (obj)=> {
  let schema = Joi.object({
    username: Joi.string().required(),
    vhost: Joi.string().required(),
    resource: Joi.string().required().allow('queue', 'exchange', 'topic'),
    name: Joi.string().required(),
    permission: Joi.string().required().allow(...Object.values(KeycloakRequests.SCOPES.resource)),
    tags: Joi.string().required()
  });
  await schema.validateAsync(obj);
};

const ResourceEndpoint = async (req, res)=> {
  try {
    await validate(req.body);
    await KeycloakRequests.authorize(req.body.username, req.body.name, req.body.permission);
    res.send('allow');    
  } catch (e) {
    res.send('deny');
  }
};
export default ResourceEndpoint;