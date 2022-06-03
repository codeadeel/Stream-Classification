# Stream Classification

* [**Introduction**](#introduction)
* [**Architecture**](#architecture)
* [**Repository Cloning**](#cloner)
* [**Results**](#convres)
* [**Container Build**](#container_build)
  * [**Single File Processor**](#single_processor)
  * [**Annotation Based File Processor**](#annotation_processor)
  * [**Trainer**](#trainer)
  * [**Stream Classification Inference Server**](#sc_infer_server)
  * [**Client: *Live Stream Local Display***](#live_stream_local)
  * [**Client: *Live Stream Browser***](#live_stream_browser)
  * [**Client: *Live Stream RTSP***](#live_stream_rtsp)
  * [**Client: *Multi Live Stream Browser***](#multi_live_stream_browser)
  * [**Client: *Multi Live Stream RTSP***](#multi_live_stream_rtsp)
  * [**Client: *Local Stream Local Display***](#local_stream_local)
  * [**Client: *Local Stream Browser***](#local_stream_browser)
  * [**Client: *Local Stream RTSP***](#local_stream_rtsp)
  
## <a name="introduction">Introduction

The subject code is responsible for Stream Classification. Final classification is concluded on the basis of temporal probabilities & implemented as server-client architecture. Currently three classes can be classified using subject archiecture.

* Advertisement
* News
* Program

The model used in this architecture is ***ConvNext-Tiny***, finetuned using ***Pytorch*** framework for Pakistani news channels.

## <a name="architexture">Architecture

The whole system is based on server-client architecture. This allows easy usablity and integration without multi accelaration devices and extra resource consumption. The macro architecture is shown below:

![Stream Classification Macro Architecture][macro_architecture]

The above macro architecture is based on following pointers:

* Everything is containerized.
* Stream Object can be processed for training data by using either:
  * Single stream object file processor ***(Recommended)***
  * Annotation based file processor
* Training pipeline with fetch data, process it and save trained model onto persistent storage.
* Output resources will be loaded by **Stream Classification Inference Server**, and server will be ready for inference.
* Multiple clients can connect to a single server for inference in ***batched mode***, using GRPC.

The micro architecture is also shown below:

![Stream Classification Micro Architecture][micro_architecture]

The above micro architecture is based on follwoing pointer:

* Everything is containerized.
* Input images will be converted to batches using batch handler.
* Batched of images will be sent to AI inference server using request handler.
* AI inference server handles batched inputs using GRPC with client identification.
* Input batch will be processed to CUDA tensor.
* Inference will be done on ***GPU / CPU*** in Pytorch.
* Classification results will be computed on the basis of temporal probabilities.
* Results are returned to respective client using request handler.
* Request handler on the client side, processes results and make it ready for utilization.

## <a name="cloner">Repository Cloning

The prerequisite for this repository is [***gdown***][gdown_link] library. It is required to download trained resources from Google Drive. The cloning and setup procedure for this repository is given below:

```bash
# To clone repository after setting up gdown
git clone https://github.com/codeadeel/Stream-Classification.git

# To download trained resources
cd ./Stream-Classification
chmod 777 ./get_resources.py
./get_resources.py
```

## <a name="convres">Results

The finetuning for the model is done on Pakistani news channel dataset. Finetuning results are given below:

***Training Epochs:*** 3  
***Learning Rate:*** 0.0001  
***Batch Size:*** 64  

***Average Training Loss:*** 0.362  
***Average Training Accuracy:*** 0.857  
***Average Validation Loss:*** 0.156  
***Average Validation Accuracy:*** 0.944  
***Testing Loss:*** 0.0306  
***Testing Accuracy:*** 0.953  

***Precision:*** 0.962  
***Recall:*** 0.952  
***F1 Score:*** 0.957  
***Jaccard Score:*** 0.9182  

Confusion Matrix after finetuning was computed as below:

![Confusion Matrix][convnext_confusion]

## <a name="container_build">Container Build

Every component of architecture is containerized, so container building container and respective execution requires certain set of commands. Following commands can be used for respective container builds:

### <a name="single_processor">Single File Processor

Single file processor is used to convert video file into training data, against given label. To build the container, following command can be used:

```bash
docker build -t stream_classification:single_processor -f Single_Processor .
```
Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -v [ Required: Your Path to Video File ]:/Video.mp4 \
    -v [ Required: Your Directory Path to Save Generated / Updated Processed Data ]:/Output \
    stream_classification:single_processor [ Your Arguments ]
```

### <a name="annotation_processor">Annotation Based File Processor

Annotation based file processor is used to convert video file into training data, against given annotation file. [***More on annotation file format***][annotation_discussion]. To build the container, following command can be used:

```bash
docker build -t stream_classification:processor -f Processor .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -v [ Required: Your Path to Video File ]:/Video.mp4 \
    -v [ Required: Your Path to Annotation File ]:/Annotation.csv \
    -v [ Required: Your Directory Path to Save Generated / Updated Processed Data ]:/Output \
    stream_classification:processor [ Your Arguments ]
```

### <a name="trainer">Trainer

Training on the subject data can be automatically done and handled by training pipeline. To build the container, following command can be used:

```bash
docker build -t stream_classification:trainer -f Build_Trainer .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it --gpus all \
    -v [ Required: Your Path to Data Directory ]:/data \
    -v [ Required / Optional: Your Directory Path to Save Output Files ]:/resources \
    stream_classification:trainer [ Your Arguments ]
```

### <a name="sc_infer_server">Stream Classification Inference Server

Stream classification inference server is responsible of batched inference, results computation and handling of clients. It can track multiple clients and multiple input modes for inference from each client. To build the container, following command can be used:

```bash
docker build -t stream_classification:server -f Build_Server .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it --gpus all \
    -v [ Optional: Your Path to Model Weights File ]:/resources/convnext.model \
    -v [ Optional: Your Path to One Hot Encoded Labels File ]:/resources/OHE.labels \
    -p [ Your Port to Expose Server ]:1234 \
    stream_classification:server [ Your Arguments ]
```

### <a name="live_stream_local">Client: ***Live Stream Local Display***

This client can be used to perform inference on live stream, and display results on local display, using .X11 socket. To build the container, following command can be used:

```bash
docker build -t stream_classification:live_client -f Live_Client .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -v [ Optional: Your Directory Path to Save Output ]:/Output \
    stream_classification:live_client [ Your Arguments ]
```

### <a name="live_stream_browser">Client: ***Live Stream Browser***

This client can be used to perform inference on live stream, and display results through webpage. To build the container, following command can be used:

```bash
docker build -t stream_classification:live_client_browser -f Live_Client_Browser .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -v [ Optional: Your Directory Path to Save Output ]:/Output \
    -p [ Optional: Your Port to Expose ]:80 \
    stream_classification:live_client_browser [ Your Arguments ]
```

### <a name="live_stream_rtsp">Client: ***Live Stream RTSP***

This client can be used to perform inference on live stream, and display results through RTSP stream. To build the container, following command can be used:

```bash
docker build -t stream_classification:live_client_rtsp -f Live_Client_RTSP .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -v [ Optional: Your Directory Path to Save Output ]:/Output \
    -p [ Optional: Your Port to Expose ]:80 \
    stream_classification:live_client_rtsp [ Your Arguments ]
```

### <a name="multi_live_stream_browser">Client: ***Multi Live Stream Browser***

This client can be used to perform inference on multiple live streams, and display results on through webpage. To build the container, following command can be used:

```bash
docker build -t stream_classification:multi_live_browser -f Multi_Live_Client_Browser .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -p [ Optional: Your Port to Expose ]:80 \
    stream_classification:multi_live_client_browser [ Your Arguments ]
```

### <a name="multi_live_stream_rtsp">Client: ***Multi Live Stream RTSP***

This client can be used to perform inference on multiple live streams, and display results through RTSP stream. To build the container, following command can be used:

```bash
docker build -t stream_classification:multi_live_rtsp -f Multi_Live_Client_RTSP .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -p [ Optional: Your Port to Expose ]:80 \
    stream_classification:multi_live_client_rtsp [ Your Arguments ]
```

### <a name="local_stream_local">Client: ***Local Stream Local Display***

This client can be used to perform inference on local video, and display results on local display, using .X11 socker. To build constainer, following command can be used:

```bash
docker build -t stream_classification:video_client -f Video_Client .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -v [ Required: Your Path to Video File]:/video.mp4 \
    -v [ Optional: Your Directory Path to Save Output ]:/Output \
    stream_classification:video_client [ Your Arguments ]
```

### <a name="local_stream_browser">Client: ***Local Stream Browser***

This client can be used to perform inference on local stream, and display results through webpage. To build the container, following command can be used:

```bash
docker build -t stream_classification:video_client_browser -f Video_Client_Browser .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -v [ Required: Your Path to Video File]:/video.mp4 \
    -v [ Optional: Your Directory Path to Save Output ]:/Output \
    -p [ Optional: Your Port to Expose ]:80 \
    stream_classification:video_client_browser [ Your Arguments ]
```

### <a name="local_stream_rtsp">Client: ***Local Stream RTSP***

This client can be used to perform inference on local stream, and display results through RTSP stream. To build the container, following command can be used:

```bash
docker build -t stream_classification:video_client_rtsp -f Video_Client_RTSP .
```

Also, to execute container, following command can be used:

```bash
docker run --rm -it \
    -v [ Required: Your Path to Video File]:/video.mp4 \
    -v [ Optional: Your Directory Path to Save Output ]:/Output \
    -p [ Optional: Your Port to Expose ]:80 \
    stream_classification:video_client_rtsp [ Your Arguments ]
```


[macro_architecture]: ./MarkDown-Data/macro_architecture.jpg
[micro_architecture]: ./MarkDown-Data/micro_architecture.jpg
[convnext_confusion]: ./MarkDown-Data/convnext_model_Confusion_Matrix.png
[annotation_discussion]: ./Trainer/README.md
[gdown_link]: https://github.com/wkentaro/gdown
