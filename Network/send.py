#!/usr/bin/env python
import pika,sys



def main(args):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='hello')
    
    channel.basic_publish(exchange='', routing_key='hello', body=' '.join(args))
    print(" [x] Sent "+''.join(args))
    connection.close()



if __name__ == '__main__':
    main(sys.argv[1:])