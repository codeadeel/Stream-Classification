# Clients Scripts

* [**Introduction**](#introduction)
* [**Client: *Live Stream Local Display***](#live_stream_local)
* [**Client: *Live Stream Browser***](#live_stream_browser)
* [**Client: *Live Stream RTSP***](#live_stream_rtsp)
* [**Client: *Multi Live Stream Browser***](#multi_live_stream_browser)
* [**Client: *Multi Live Stream RTSP***](#multi_live_stream_rtsp)
* [**Client: *Local Stream Local Display***](#local_stream_local)
* [**Client: *Local Stream Browser***](#local_stream_browser)
* [**Client: *Local Stream RTSP***](#local_stream_rtsp)

## <a name="introduction">Introduction

Clients scripts are used for multiple types of inference from stream classification inference server. Clients can be of many types and can server multiple purposes. Each client is identified by inference server as different client, as ID for each client is different. This helps inference server to refine output for every client. Multiple clients can be build and containerized. Some of the most basic client are discussed here.

## <a name="live_stream_local">Client: ***Live Stream Local Display***

This client can be used to perform inference on live stream, and display results on local display, using .X11 socket. This [script][lsld] take following arguments as input:

```bash
usage: live_client.py [-h] -l LINK -ip SERVER_IP [-sk SKIP] [-tz TIME_ZONE]
                      [-fmpg FFMPEG] [-ytdl YOUTUBE_DL]

Stream Classification Live Inference Client.

optional arguments:
  -h, --help            show this help message and exit
  -l, --link            Live Data Link
  -ip, --server_ip      IP Address to GRPC Server => IP:Port
  -sk, --skip           Number of Frames to Skip in Live Inference
  -tz, --time_zone      Current Time Zone
  -fmpg, --ffmpeg       Absolute Address to Standalone FFMPEG File
  -ytdl, --youtube_dl   Absolute Address to Standalone Youtube-DL File
```

## <a name="live_stream_browser">Client: ***Live Stream Browser***

This client can be used to perform inference on live stream, and display results through webpage. This [script][lsb] take following arguments as input:

```bash
usage: live_client_browser.py [-h] -l LINK -ip SERVER_IP [-cip CLIENT_IP]
                              [-sk SKIP] [-tz TIME_ZONE] [-fmpg FFMPEG]
                              [-ytdl YOUTUBE_DL]

Stream Classification Browser Based Live Inference Client.

optional arguments:
  -h, --help            show this help message and exit
  -l, --link            Live Data Link
  -ip, --server_ip      IP Address to GRPC Server => IP:Port
  -cip, --client_ip     IP Address for Client => IP:Port
  -sk, --skip           Number of Frames to Skip in Live Inference
  -tz, --time_zone      Current Time Zone
  -fmpg, --ffmpeg       Absolute Address to Standalone FFMPEG File
  -ytdl, --youtube_dl   Absolute Address to Standalone Youtube-DL File
```

## <a name="live_stream_rtsp">Client: ***Live Stream RTSP***

This client can be used to perform inference on live stream, and display results through RTSP stream. This [script][lsr] take following arguments as input:

```bash
usage: live_client_rtsp.py [-h] -l LINK -ip SERVER_IP [-cip CLIENT_IP]
                           [-sk SKIP] [-tz TIME_ZONE] [-fmpg FFMPEG]
                           [-ytdl YOUTUBE_DL] [-tw TARGET_WIDTH]
                           [-th TARGET_HEIGHT]

Stream Classification RTSP Based Live Inference Client.

optional arguments:
  -h, --help            show this help message and exit
  -l, --link            Live Data Link
  -ip, --server_ip      IP Address to GRPC Server => IP:Port
  -cip, --client_ip     IP Address for Client => IP:Port
  -sk, --skip SKIP      Number of Frames to Skip in Live Inference
  -tz, --time_zone      Current Time Zone
  -fmpg, --ffmpeg       Absolute Address to Standalone FFMPEG File
  -ytdl, --youtube_dl   Absolute Address to Standalone Youtube-DL File
  -tw, --target_width   Target Frame Width
  -th, --target_height  Target Frame Height
```

## <a name="multi_live_stream_browser">Client: ***Multi Live Stream Browser***

This client can be used to perform inference on multiple live streams, and display results on through webpage. This [script][mlsb] take following arguments as input:

```bash
usage: multi_live_client_browser.py [-h] -l LINK [LINK ...] -ip SERVER_IP
                                    [-cip CLIENT_IP] [-sk SKIP]
                                    [-tz TIME_ZONE] [-fmpg FFMPEG]
                                    [-ytdl YOUTUBE_DL]

Stream Classification Browser Based Live Inference Client.

optional arguments:
  -h, --help                        show this help message and exit
  -l [LINK ...], --link [LINK ...]  Live Data Link List
  -ip, --server_ip                  IP Address to GRPC Server => IP:Port
  -cip, --client_ip                 IP Address for Client => IP:Port
  -sk, --skip                       Number of Frames to Skip in Live Inference
  -tz, --time_zone                  Current Time Zone
  -fmpg, --ffmpeg                   Absolute Address to Standalone FFMPEG File
  -ytdl, --youtube_dl               Absolute Address to Standalone Youtube-DL File
```

## <a name="multi_live_stream_rtsp">Client: ***Multi Live Stream RTSP***

This client can be used to perform inference on multiple live streams, and display results through RTSP stream. This [script][mlsr] take following arguments as input:

```bash
usage: multi_live_client_rtsp.py [-h] -l LINK [LINK ...] -ip SERVER_IP
                                 [-cip CLIENT_IP] [-sk SKIP] [-tz TIME_ZONE]
                                 [-fmpg FFMPEG] [-ytdl YOUTUBE_DL]
                                 [-tw TARGET_WIDTH] [-th TARGET_HEIGHT]

Stream Classification RTSP Based Live Inference Client.

optional arguments:
  -h, --help                        show this help message and exit
  -l [LINK ...], --link [LINK ...]  Live Data Link List
  -ip, --server_ip                  IP Address to GRPC Server => IP:Port
  -cip, --client_ip                 IP Address for Client => IP:Port
  -sk, --skip                       Number of Frames to Skip in Live Inference
  -tz, --time_zone                  Current Time Zone
  -fmpg, --ffmpeg                   Absolute Address to Standalone FFMPEG File
  -ytdl, --youtube_dl               Absolute Address to Standalone Youtube-DL File
  -tw, --target_width               Target Frame Width
  -th, --target_height              Target Frame Height
```

## <a name="local_stream_local">Client: ***Local Stream Local Display***

This client can be used to perform inference on local video, and display results on local display, using .X11 socker. This [script][losld] take following arguments as input:

```bash
usage: video_client.py [-h] -ip SERVER_IP [-l LINK] [-sk SKIP] [-fmpg FFMPEG]

Stream Classification Local Video Inference Client.

optional arguments:
  -h, --help            show this help message and exit
  -ip, --server_ip      IP Address to GRPC Server => IP:Port
  -l, --link            Local Video Address
  -sk, --skip           Number of Frames to Skip in Local Video Inference
  -fmpg, --ffmpeg       Absolute Address to Standalone FFMPEG File
```

## <a name="local_stream_browser">Client: ***Local Stream Browser***

This client can be used to perform inference on local stream, and display results through webpage. This [script][losb] take following arguments as input:

```bash
usage: video_client_browser.py [-h] -ip SERVER_IP [-cip CLIENT_IP] [-l LINK]
                               [-sk SKIP]

Stream Classification Browser Based Local Video Inference Client.

optional arguments:
  -h, --help            show this help message and exit
  -ip, --server_ip      IP Address to GRPC Server => IP:Port
  -cip, --client_ip     IP Address for Client => IP:Port
  -l, --link            Local Video Address
  -sk, --skip           Number of Frames to Skip in Local Video Inference
```

## <a name="local_stream_rtsp">Client: ***Local Stream RTSP***

This client can be used to perform inference on local stream, and display results through RTSP stream. This [script][losr] take following arguments as input:

```bash
usage: video_client_rtsp.py [-h] -ip SERVER_IP [-cip CLIENT_IP] [-l LINK]
                            [-sk SKIP] [-tw TARGET_WIDTH] [-th TARGET_HEIGHT]

Stream Classification RTSP Based Local Video Inference Client.

optional arguments:
  -h, --help            show this help message and exit
  -ip, --server_ip      IP Address to GRPC Server => IP:Port
  -cip, --client_ip     IP Address for Client => IP:Port
  -l, --link            Local Video Address
  -sk, --skip           Number of Frames to Skip in Local Video Inference
  -tw, --target_width   Target Frame Width
  -th, --target_height  Target Frame Height
```

[lsld]: ./live_client.py
[lsb]: ./live_client_browser.py
[lsr]: ./live_client_rtsp.py
[mlsb]: ./multi_live_client_browser.py
[mlsr]: ./multi_live_client_rtsp.py
[losld]: ./video_client.py
[losb]: ./video_client_browser.py
[losr]: ./video_client_rtsp.py
