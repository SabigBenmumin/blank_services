import pika
from blank import calculator
# import os
import time

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost')
)
channel = connection.channel()
channel.queue_declare(queue='rpc_queue')

def on_request(ch, method, props, body):
    filepaths = body.decode('utf-8')
    source_path, target_path, grid_size, task_id = filepaths.split(",")
    print(f" [.] Received filepaths")
    start = time.time()
    print(f"\tsrc: {source_path}")
    print(f"\ttgt: {target_path}")
    print(f"\tgrid_size: {grid_size}")
    volume_change, sand_increase, sand_decrease = calculator(source_path, target_path, task_id, float(grid_size))
    # print(f"\tCalculating volume change: {calculator(source_path, target_path)}")
    print(f"\tCalculating volume change: {volume_change}")
    # response = f" [.] Received {source_path} and {target_path}",volume_change
    # response = str(volume_change)+","+ result_plot_path
    response = f'{volume_change},{sand_increase},{sand_decrease}'

    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(correlation_id = props.correlation_id),
        body=response
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)
    acktime = time.time()
    elapsed_time = acktime - start
    print(f"\tWorker takes {elapsed_time} seconds to cook.")
    # os.remove(source_path)
    # os.remove(target_path)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='rpc_queue', on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()