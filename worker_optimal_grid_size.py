import pika
import time
from blank import get_optimal_gridsize

def on_optimal_grid_size_request(ch, method, properties, body):
    try:
        filepaths = body.decode('utf-8')
        source_path, target_path, task_id = filepaths.split(",")
        print(f" [.] Received filepaths")
        start = time.time()
        print(f"\tsrc: {source_path}")
        print(f"\ttgt: {target_path}")

        
        optimal_grid_size = get_optimal_gridsize(source_path, target_path)
        print(f"reponse optimal grid size: {optimal_grid_size}")
        response = str(optimal_grid_size)
        
        ch.basic_publish(
            exchange="",
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(
                correlation_id=properties.correlation_id
            ),
            body=response
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"error ja: {e}")

def start_optimal_grid_size_worker():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost", heartbeat=600, blocked_connection_timeout=300))
    channel = connection.channel()
    channel.queue_declare(queue="rpc_queue_optimal_grid_size")

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="rpc_queue_optimal_grid_size", on_message_callback=on_optimal_grid_size_request)

    print("Waiting for requests to calculate optimal grid size...")
    channel.start_consuming()

if __name__ == "__main__":
    start_optimal_grid_size_worker()
