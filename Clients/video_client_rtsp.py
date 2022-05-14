#!/usr/bin/env python3

"""
STREAM CLASSIFICATION LOCAL INFERENCE CLIENT FOR RTSP
=====================================================

The following program is used to perform local data inference on the subject data & display results through RTSP
"""

# %%
# Importing Libraries
from inference_client import *
import argparse
import cv2 as ocv
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

# %%
# Main Local Video Client Inference Class
class Video:
    def __init__(self, addr, server_ip, skip=0):
        """
        This method is used to initialize Stream Classification local video inference client

        Method Input
        =============
        addr : Local address to data
        server_ip : Server IP at which GRPC server is running
                            Format : "IP:Port"
                            Example : '0.0.0.0:1234'
        skip : Number of frames to skip in live inference ( default : 0 )

        Method Output
        ==============
        None
        """
        self.vid_addr = addr
        self.sc_server_ip = server_ip
        self.__color__ = (0, 0, 255)
        self.skip = skip
        self.ret = True
        self.skip_count = self.skip
        self.count = 0
        self.output_saving_addr = '/Output'
        self.__vid_obj__ = ocv.VideoCapture(self.vid_addr)
        self.FPS = int(self.__vid_obj__.get(ocv.CAP_PROP_FPS))
        self.height = int(self.__vid_obj__.get(ocv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.__vid_obj__.get(ocv.CAP_PROP_FRAME_WIDTH))
        self.__total_frames__ = int(self.__vid_obj__.get(ocv.CAP_PROP_FRAME_COUNT))
        self.stream_inference = sc_client(self.sc_server_ip)
        self.csv_str = 'Frame #,Stream Classification\n'
    
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
        ===========================================================
        | Stream Classification RTSP Based Local Inference Client |
        ===========================================================
        """)
        print(f'Stream Classification Server IP Address: {self.sc_server_ip}')
        print(f'Client ID: {self.stream_inference.client_name}')
        print(f'Video FPS: {self.FPS}')
        print(f'Video Height: {self.height}')
        print(f'Video Width: {self.width}')
        print(f'Total Video Frames: {self.__total_frames__}')
        print(f'Number of Frames to Skip During Inference: {self.skip}')
        print(f'Output Data Saving Directory: {self.output_saving_addr}')
        print('\n---------------------------------------------')
        print('>>>>> Press CTRL+C to End Stream')
        return '\n'
    
    def __call__(self):
        """
        This method is used to handle & perform inference on local data

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        while self.ret:
            self.ret, cv_dat = self.__vid_obj__.read()
            if not self.ret:
                break
            if self.skip_count < self.skip:
                self.skip_count += 1
            else:
                self.skip_count = 0
                pil_dat = Image.fromarray(ocv.cvtColor(cv_dat, ocv.COLOR_BGR2RGB))
                self.res = self.stream_inference([pil_dat])
                self.csv_str += f'{self.count},{self.res[0][0]}\n'
                with open(f'/{self.output_saving_addr}/Output_Metadata.csv', 'w') as file1:
                    file1.write(self.csv_str)
            cv_dat = ocv.putText(cv_dat, self.res[0][0], (50, 50), ocv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, ocv.LINE_AA)
            cv_dat = ocv.putText(cv_dat, self.res[0][0], (50, 50), ocv.FONT_HERSHEY_SIMPLEX, 1, self.__color__, 1, ocv.LINE_AA)
            self.count += 1
            yield cv_dat

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
        self.vid_iter = vid()
        self.number_frames = 0
        self.duration = 1 / vid.FPS * Gst.SECOND
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(args['target_width'], args['target_height'], vid.FPS)
    
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
# RTSP Based Local Video Client Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stream Classification RTSP Based Local Video Inference Client.')
    parser.add_argument('-ip', '--server_ip', type = str, help = 'IP Address to GRPC Server => IP:Port', required = True)
    parser.add_argument('-cip', '--client_ip', type = str, help = 'IP Address for Client => IP:Port', default = '0.0.0.0:80/video')
    parser.add_argument('-l', '--link', type = str, help = 'Local Video Address', default = '/video.mp4' )
    parser.add_argument('-sk', '--skip', type = int, help = 'Number of Frames to Skip in Local Video Inference', default = 0)
    parser.add_argument('-tw', '--target_width', type = int, help = 'Target Frame Width', default = 1920)
    parser.add_argument('-th', '--target_height', type = int, help = 'Target Frame Height', default = 1080)
    args = vars(parser.parse_args())
    vid = Video(args['link'], args['server_ip'], skip = args['skip'])
    print(vid)
    cip_ip = args['client_ip'].split(':')[0]
    cip_port = args['client_ip'].split(':')[1].split('/')[0]
    cip_addr = args['client_ip'].split(':')[1].split('/')[1]
    print(f'RTSP Link: rtsp://{cip_ip}:{cip_port}/{cip_addr}\n')
    GObject.threads_init()
    Gst.init(None)
    server = GstServer()
    loop = GObject.MainLoop()
    loop.run()
        