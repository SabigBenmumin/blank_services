import pika
import time
from blank import calculator

def on_volume_change_request(ch, method, properties, body):
    # ตรงนี้คือที่คุณจะทำการคำนวณ volume change ตามข้อมูลที่ได้รับ
    filepaths = body.decode('utf-8')
    source_path, target_path, grid_size, task_id = filepaths.split(",")
    print(f" [.] Received filepaths")
    start = time.time()
    print(f"\tsrc: {source_path}")
    print(f"\ttgt: {target_path}")
    print(f"\tgrid_size: {grid_size}")
    volume_change, sand_increase, sand_decrease = calculator(source_path, target_path, task_id, float(grid_size))
    print(f"\tCalculating volume change: {volume_change}")

    response = f"{volume_change},{sand_increase},{sand_decrease}"
    
    ch.basic_publish(
        exchange="",
        routing_key=properties.reply_to,
        properties=pika.BasicProperties(
            correlation_id=properties.correlation_id
        ),
        body=response
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)
    acktime = time.time()
    elapsed_time = acktime - start
    print(f"\tWorker takes {elapsed_time} seconds to cook.")

def start_volume_change_worker():
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host="localhost",
        heartbeat=600,
        blocked_connection_timeout=300
        )
    )
    channel = connection.channel()
    channel.queue_declare(queue="rpc_queue_volume_change")

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="rpc_queue_volume_change", on_message_callback=on_volume_change_request)

    print("Waiting for requests to calculate volume change...")
    channel.start_consuming()

if __name__ == "__main__":
    start_volume_change_worker()
