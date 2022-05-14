# Server Scripts

* [**Introduction**](#introduction)
* [**Stream Classification Inference Server**](#sc_infer_server)
* [**Stream Classification Inference Client**](#sc_infer_client)

## <a name="introduction">Introduction

Server scripts are used to power server client architecture. Server scripts are responsible for inference handling and client identification. Server architecture makes it possible to load trained resources automatically for inference, and listens for client for request. Server client architecture communication is based on GRPC.

## <a name="sc_infer_server">Stream Classification Inference Server

Stream classification inference server is responsible of batched inference, results computation and handling of clients. It can track multiple clients and multiple input modes for inference from each client. This [script][ins] take following arguments as input:

```bash
usage: inference_server.py [-h] [-scw SC_WINDOW] [-ip SERVER_IP]
                           [-msg MSG_LEN] [-wrk WORKERS] [-ohe OHE]
                           [-la LADDR]

Stream Classification Inference Server.

optional arguments:
  -h, --help            show this help message and exit
  -scw, --sc_window     Stream Classification Averaging Window Size
  -ip, --server_ip      IP Address to Start GRPC Server
  -msg, --msg_len       Message Length Subject to Communication by GRPC
  -wrk, --workers       Number of Workers to Used by GRPC
  -ohe, --OHE           Absolute Address of One Hot Encoded Labels File
  -la, --laddr          Absolute Address of Model File
```

## <a name="sc_infer_client">Stream Classification Inference Client

Client scripts are used to request images for inference to server. In simple words, they send batch of images to inference server, and return results after process. Usage for this [script][inc] is given as following:

```python
from inference_client import *

stream_classification_inference_server_ip = '172.17.0.2:1234'

img1 = Image.open('some image path')
img2 = Image.open('some image path')

img_batch = [img1, img2, ...]

target_server = sc_client(stream_classification_inference_server_ip)

results = target_server(img_batch)

```

[ins]: ./inference_server.py
[inc]: ./inference_client.py
