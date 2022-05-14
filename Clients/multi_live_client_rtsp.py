#!/usr/bin/env python3

"""
STREAM CLASSIFICATION MULTI LIVE INFERENCE CLIENT FOR RTSP
==========================================================

The following program is used to perform multi live data inference on the subject data & display results through RTSP
"""

# %%
# Importing Libraries
from inference_client import *
import argparse
import pytz
import datetime
import subprocess as sp
import cv2 as ocv
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

# %%
# Main Live Client Inference Class
class Live:
    def __init__(self, addr, server_ip, tzone='Asia/Karachi', ytdl='/resources/youtube-dl', ffmpeg='/resources/ffmpeg', skip=0):
        """
        This method is used to initialize Stream Classification live inference client

        Method Input
        =============
        addr : List of live data links
        server_ip : Server IP at which GRPC server is running
                            Format : "IP:Port"
                            Example : '0.0.0.0:1234'
        tzone : Time zone for the time to record on ( default : Asia/Karachi )
        ytdl : Absolute address of Youtube-DL standalone file
        ffmpeg : Absolute address of FFMPEG standalone file
        skip : Number of frames to skip in live inference ( default : 0 )

        Method Output
        ==============
        None
        """
        self.live_links = addr
        self.sc_server_ip = server_ip
        self.ytdl_addr = ytdl
        self.fmpg_addr = ffmpeg
        self.skip = skip
        self.time_zone = tzone
        self.__quality__ = 94
        self.__dim__ = (480, 854, 3)
        self.__color__ = (0, 0, 255)
        self.count = 0
        self.skip_count = self.skip
        self.FPS = 30
        self.stream_inference = sc_client(self.sc_server_ip)
        self.streams = list()
        for i in self.live_links:
            stream = sp.check_output(['python3', self.ytdl_addr, '-f', str(self.__quality__), '--get-url', i]).decode('utf-8')
            stream = sp.Popen([self.fmpg_addr, "-i", stream, "-loglevel", "quiet", "-an", "-f", "image2pipe", "-s", str(self.__dim__[1]) + 'x' + str(self.__dim__[0]), "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"], stdin=sp.PIPE, stdout=sp.PIPE)
            self.streams.append(stream)
        print('>>>>> All Streams Captured')
        
    def __str__(self):
        """
        This method is __str__ implementation of subject class

        Method Input
        =============
        None

        Method Output
        ==============
        New Line
        """
        print("""
        ================================================================
        | Stream Classification RTSP Based Multi Live Inference Client |
        ================================================================
        """)
        print(f'Live Links: {self.live_links}')
        print(f'Stream Classification Server IP Address: {self.sc_server_ip}')
        print(f'Client ID: {self.stream_inference.client_name}')
        print(f'Time Zone: {self.time_zone}')
        print(f'Number of Frames to Skip During Inference: {self.skip}')
        print(f'YouTube-DL Quality: {self.__quality__}')
        print('\n---------------------------------------------')
        print('>>>>> Press Ctrl+C to End Stream')
        return '\n'
        
    def __get_data__(self):
        """
        This method is used to fetch frame from live data

        Method Input
        =============
        None

        Method Output
        ==============
        Returns frame as tuple of OpenCV & Pillow image format
                            ( OpenCV Image List, Pillow Image List )
        """
        inf_frames, disp_frames = list(), list()
        for i in self.streams:
            frame = np.frombuffer(i.stdout.read(self.__dim__[1] * self.__dim__[0] * self.__dim__[2]), dtype='uint8').reshape((self.__dim__[0], self.__dim__[1], self.__dim__[2]))
            disp_frames.append(frame)
            inf_frames.append(Image.fromarray(ocv.cvtColor(frame, ocv.COLOR_BGR2RGB)))
        return disp_frames, inf_frames

    def __call__(self):
        """
        This method is used to handle & perform inference on live data

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        while True:
            td = datetime.datetime.now().astimezone(pytz.timezone(self.time_zone))
            cv_dat, pil_dat = self.__get_data__()
            if self.skip_count < self.skip:
                self.skip_count += 1
            else:
                self.skip_count = 0
                self.res = self.stream_inference(pil_dat)
            for i in range(len(cv_dat)):
                cv_dat[i] = ocv.putText(cv_dat[i], self.res[0][i], (50, 50), ocv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, ocv.LINE_AA)
                cv_dat[i] = ocv.putText(cv_dat[i], self.res[0][i], (50, 50), ocv.FONT_HERSHEY_SIMPLEX, 1, self.__color__, 1, ocv.LINE_AA)
                cv_dat[i] = ocv.putText(cv_dat[i], '{}-{}-{} {}:{}:{}'.format(str(td.day).zfill(2), str(td.month).zfill(2), td.year, str(td.hour).zfill(2), str(td.minute).zfill(2), str(td.second).zfill(2)), (50, 85), ocv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, ocv.LINE_AA)
                cv_dat[i] = ocv.putText(cv_dat[i], '{}-{}-{} {}:{}:{}'.format(str(td.day).zfill(2), str(td.month).zfill(2), td.year, str(td.hour).zfill(2), str(td.minute).zfill(2), str(td.second).zfill(2)), (50, 85), ocv.FONT_HERSHEY_SIMPLEX, 1, self.__color__, 1, ocv.LINE_AA)
            counter, out_img = 0, list()
            template = np.reshape(np.arange(((len(cv_dat)//3)+1)*3), (-1, 3))
            if len(cv_dat)%3 == 0:
                template = template[:-1]
            for i in template:
                temp_list = list()
                for j in i:
                    try:
                        curr_img = cv_dat[counter]
                        counter += 1
                    except:
                        curr_img = np.zeros_like(cv_dat[0], dtype=np.uint8)
                    temp_list.append([curr_img])
                out_img.append(temp_list)
            out_img = np.block(out_img)
            yield out_img

# %%
# RTSP Server
class Video_Server(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        """
        This method is used to initialize RTSP server

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        super(Video_Server, self).__init__(**properties)
        self.vid_iter = liv()
        self.number_frames = 0
        self.duration = 1 / liv.FPS * Gst.SECOND
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(args['target_width'], args['target_height'], liv.FPS)
    
    def on_need_data(self, src, length):
        """
        This method is used to get the processed frame & push data to RTSP server

        Method Input
        =============
        RTSP class arguments

        Method Output
        ==============
        None
        """
        frame = ocv.resize(next(self.vid_iter), (args['target_width'], args['target_height']), interpolation = ocv.INTER_LINEAR)
        data = frame.tostring()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = int(timestamp)
        buf.offset = timestamp
        self.number_frames += 1
        retval = src.emit('push-buffer', buf)
    
    def do_create_element(self, url):
        """
        This method is used to pasrse launch string to push data

        Method Input
        =============
        RTSP class arguments

        Method Output
        ==============
        None
        """
        return Gst.parse_launch(self.launch_string)
    
    def do_configure(self, rtsp_media):
        """
        This method is used for internal configuration of RTSP server

        Method Input
        =============
        RTSP class arguments

        Method Output
        ==============
        None
        """
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

# %%
# Gstreamer Server
class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        """
        This method is used to initialize Gstreamer server

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        super(GstServer, self).__init__(**properties)
        self.factory = Video_Server()
        self.factory.set_shared(True)
        self.set_service(cip_port)
        self.get_mount_points().add_factory(f'/{cip_addr}', self.factory)
        self.attach(None)

# %%
# RTSP Based Live Client Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stream Classification RTSP Based Live Inference Client.')
    parser.add_argument('-l', '--link', nargs = '+', help = 'Live Data Link List', required = True)
    parser.add_argument('-ip', '--server_ip', type = str, help = 'IP Address to GRPC Server => IP:Port', required = True)
    parser.add_argument('-cip', '--client_ip', type = str, help = 'IP Address for Client => IP:Port', default = '0.0.0.0:80/video')
    parser.add_argument('-sk', '--skip', type = int, help = 'Number of Frames to Skip in Live Inference', default = 0)
    parser.add_argument('-tz', '--time_zone', type = str, help = 'Current Time Zone', default = 'Asia/Karachi')
    parser.add_argument('-fmpg', '--ffmpeg', type = str, help ='Absolute Address to Standalone FFMPEG File', default = '/resources/ffmpeg')
    parser.add_argument('-ytdl', '--youtube_dl', type = str, help ='Absolute Address to Standalone Youtube-DL File', default = '/resources/youtube-dl')
    parser.add_argument('-tw', '--target_width', type = int, help = 'Target Frame Width', default = 1920)
    parser.add_argument('-th', '--target_height', type = int, help = 'Target Frame Height', default = 1080)
    args = vars(parser.parse_args())
    liv = Live(args['link'], args['server_ip'], tzone = args['time_zone'], ytdl = args['youtube_dl'], ffmpeg = args['ffmpeg'], skip = args['skip'])
    print(liv)
    cip_ip = args['client_ip'].split(':')[0]
    cip_port = args['client_ip'].split(':')[1].split('/')[0]
    cip_addr = args['client_ip'].split(':')[1].split('/')[1]
    print(f'RTSP Link: rtsp://{cip_ip}:{cip_port}/{cip_addr}\n')
    GObject.threads_init()
    Gst.init(None)
    server = GstServer()
    loop = GObject.MainLoop()
    loop.run()
