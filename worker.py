import pika
from blank import calculator

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost')
)
channel = connection.channel()
channel.queue_declare(queue='rpc_queue')

def on_request(ch, method, props, body):
    filepaths = body.decode('utf-8')
    source_path, target_path = filepaths.split(",")
    print(f" [.] Received filepaths")
    print(f"\tsrc: {source_path}")
    print(f"\ttgt: {target_path}")
    volume_change = calculator(source_path, target_path)
    # print(f"\tCalculating volume change: {calculator(source_path, target_path)}")
    print(f"\tCalculating volume change: {volume_change}")
    # response = f" [.] Received {source_path} and {target_path}",volume_change
    response = str(volume_change)

    ch.basic_publish(exchange='',
                        routing_key=props.reply_to,
                        properties=pika.BasicProperties(correlation_id = props.correlation_id),
                        body=response
                    )
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()