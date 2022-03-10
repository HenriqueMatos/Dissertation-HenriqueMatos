#!/usr/bin/env python
import pika
import sys


def main(args):
    credentials = pika.PlainCredentials('admin', 'admin')

    connection_parameters = pika.ConnectionParameters(
        'localhost', credentials=credentials, virtual_host="keycloak_test")
    connection = pika.BlockingConnection(
        connection_parameters)
    channel = connection.channel()

    channel.queue_declare(queue='hello')

    channel.basic_publish(
        exchange='', routing_key='hello', body=' '.join(args))
    print(" [x] Sent "+''.join(args))
    connection.close()


if __name__ == '__main__':
    main(sys.argv[1:])
