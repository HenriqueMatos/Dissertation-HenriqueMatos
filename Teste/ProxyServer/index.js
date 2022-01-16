// import Express                          from 'express';
// import BodyParser                       from 'body-parser';
// import { User, Vhost, Resource, Topic } from './endpoints';

const express = require('express')
const app = express()
var BodyParser = require('body-parser')
const port = 3000;

app.use(BodyParser.text({ type: 'text/html', limit: '1mb' }));
app.use(BodyParser.urlencoded({ extended: true, limit: '1mb' }));
app.use(BodyParser.json({ limit: '1mb' }));


const User = require('./user');

app.post('/user', User);
app.post('/vhost', Vhost);
app.post('/resource', Resource);
app.post('/topic', Topic);

app.listen(port, () => {
  console.log(`RabbitMQ proxy listening on port ${port}`);
});