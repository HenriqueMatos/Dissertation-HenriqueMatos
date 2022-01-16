
const Joi = require('@hapi/joi')
const KeycloakRequests = require('./keycloak_requests')
// import KeycloakRequests from './keycloak_requests';

const validate = async (obj)=> {
  let schema = Joi.object({
    username: Joi.string().required(),
    password: Joi.string().required()
  });
  await schema.validateAsync(obj);
} 

const UserEndpoint = async (req, res)=> {
  try {
    await validate(req.body);
    let authResponse = await KeycloakRequests.authenticate(req.body.username, req.body.password);
    res.send(authResponse);
  } catch(e) {
    res.send('deny');
  }
};  
export default UserEndpoint;