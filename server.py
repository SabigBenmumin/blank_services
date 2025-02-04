import pika
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
from fastapi.middleware.cors import CORSMiddleware

UPLOAD_DIR = Path("uploads")

class RPCClient(object):
    def __init__(self):
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.response = None
        self.corr_id = None
        self.connect()

    def connect(self):
        while True:
            try:
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters("localhost")
                )
                self.channel = self.connection.channel()

                result = self.channel.queue_declare(queue="", exclusive=True)
                self.callback_queue = result.method.queue

                self.channel.basic_consume(
                    queue=self.callback_queue,
                    on_message_callback=self.on_response,
                    auto_ack=True
                )
                break
            except pika.exceptions.AMQPConnectionError:
                print("Connection failed, retrying in 5 seconds...")
                time.sleep(5)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, path):
        if self.connection.is_closed or self.channel.is_closed:
            self.connect()
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange="",
            routing_key="rpc_queue",
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id
            ),
            body=path
        )
        while self.response is None:
            self.connection.process_data_events(time_limit=None)
        return self.response.decode('utf-8')

rpc_client = RPCClient()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_event():
    rpc_client.connection.close()

@app.post("/uploadfiles/")
async def create_upload_file(
        source_file: UploadFile = File(...), 
        target_file: UploadFile = File(...),
        grid_size: float = Form(0.1)
    ):
    try:
        source_file_path = UPLOAD_DIR / source_file.filename
        target_file_path = UPLOAD_DIR / target_file.filename
        with source_file_path.open("wb") as buffer:
            shutil.copyfileobj(source_file.file, buffer)
        with target_file_path.open("wb") as buffer:
            shutil.copyfileobj(target_file.file, buffer)
        
        response = rpc_client.call(f"{str(source_file_path)},{str(target_file_path)},{grid_size}")
        return JSONResponse(content={
            "files": (source_file.filename, target_file.filename), 
            "status": "file uploaded successfully", 
            "worker_response_volumeChange": response
        })
    except Exception as e:
        return JSONResponse(content={
            "files": (source_file.filename, target_file.filename), 
            "status": "file upload failed, please try again",
            "error": str(e)
        })