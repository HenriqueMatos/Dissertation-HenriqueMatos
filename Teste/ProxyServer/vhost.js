import Joi              from '@hapi/joi';
import KeycloakRequests from './keycloak_requests';

const validate = async (obj)=> {
  let schema = Joi.object({
    username: Joi.string().required(),
    vhost: Joi.string().required(),
    ip: Joi.string().required(),
    tags: Joi.string().required()
  });
  await schema.validateAsync(obj);
} 

const VhostEndpoint = async (req, res)=> {
  try {
    await validate(req.body);
    await KeycloakRequests.authorize(req.body.username, req.body.vhost, KeycloakRequests.SCOPES.vhost.access);
    res.send('allow');    
  } catch (e) {
    res.send('deny');
  }
};
export default VhostEndpoint;