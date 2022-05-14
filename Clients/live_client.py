#!/usr/bin/env python3

"""
STREAM CLASSIFICATION LIVE INFERENCE CLIENT
===========================================

The following program is used to perform live data inference on the subject data
"""

# %%
# Importing Libraries
from inference_client import *
import os
import argparse
import pytz
import datetime
import subprocess as sp
import cv2 as ocv

# %%
# Main Live Client Inference Class
class Live:
    def __init__(self, addr, server_ip, tzone='Asia/Karachi', ytdl='/resources/youtube-dl', ffmpeg='/resources/ffmpeg'):
        """
        This method is used to initialize Stream Classification live inference client

        Method Input
        =============
        addr : Link to live data
        server_ip : Server IP at which GRPC server is running
                            Format : "IP:Port"
                            Example : '0.0.0.0:1234'
        tzone : Time zone for the time to record on ( default : Asia/Karachi )
        ytdl : Absolute address of Youtube-DL standalone file
        ffmpeg : Absolute address of FFMPEG standalone file

        Method Output
        ==============
        None
        """
        self.live_link = addr
        self.sc_server_ip = server_ip
        self.ytdl_addr = ytdl
        self.fmpg_addr = ffmpeg
        self.time_zone = tzone
        self.__quality__ = 94
        self.__dim__ = (480, 854, 3)
        self.__color__ = (0, 0, 255)
        self.__out_framerate__ = 30
        self.output_saving_addr = '/Output'
        self.stream_inference = sc_client(self.sc_server_ip)
        if not os.path.exists('/Frames'):
            os.mkdir('/Frames')
        self.stream = sp.check_output(['python3', self.ytdl_addr, '-f', str(self.__quality__), '--get-url', addr]).decode('utf-8')
        self.stream = sp.Popen([self.fmpg_addr, "-i", self.stream, "-loglevel", "quiet", "-an", "-f", "image2pipe", "-s", str(self.__dim__[1]) + 'x' + str(self.__dim__[0]), "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"], stdin=sp.PIPE, stdout=sp.PIPE)
    
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
        ===============================================
        | Stream Classification Live Inference Client |
        ===============================================
        """)
        print(f'Live Link: {self.live_link}')
        print(f'Stream Classification Server IP Address: {self.sc_server_ip}')
        print(f'Client ID: {self.stream_inference.client_name}')
        print(f'Time Zone: {self.time_zone}')
        print(f'YouTube-DL Quality: {self.__quality__}')
        print(f'Output Frame Rate: {self.__out_framerate__}')
        print(f'Output Data Saving Directory: {self.output_saving_addr}')
        print('\n---------------------------------------------')
        print('>>>>> Press q to End Stream')
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
                            ( OpenCV Image, Pillow Image )
        """
        frame = np.frombuffer(self.stream.stdout.read(self.__dim__[1] * self.__dim__[0] * self.__dim__[2]), dtype='uint8').reshape((self.__dim__[0], self.__dim__[1], self.__dim__[2]))
        return frame, Image.fromarray(ocv.cvtColor(frame, ocv.COLOR_BGR2RGB))
    
    def __call__(self, skip = 0):
        """
        This method is used to handle & perform inference on live data

        Method Input
        =============
        skip : Number of frames to skip in live inference ( default : 0 )

        Method Output
        ==============
        None
        """
        count, skip_count = 0, skip
        csv_str = 'Date,Time,Frame #,Stream Classification\n'
        while True:
            td = datetime.datetime.now().astimezone(pytz.timezone(self.time_zone))
            cv_dat, pil_dat = self.__get_data__()
            if skip_count < skip:
                skip_count += 1
            else:
                skip_count = 0
                res = self.stream_inference([pil_dat])
                csv_str += f'{str(td.day).zfill(2)}-{str(td.month).zfill(2)}-{td.year},{str(td.hour).zfill(2)}:{str(td.minute).zfill(2)}:{str(td.second).zfill(2)},{count},{res[0][0]}\n'
                with open(f'/{self.output_saving_addr}/Output_Metadata.csv', 'w') as file1:
                    file1.write(csv_str)
            cv_dat = ocv.putText(cv_dat, res[0][0], (50, 50), ocv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, ocv.LINE_AA)
            cv_dat = ocv.putText(cv_dat, res[0][0], (50, 50), ocv.FONT_HERSHEY_SIMPLEX, 1, self.__color__, 1, ocv.LINE_AA)
            cv_dat = ocv.putText(cv_dat, '{}-{}-{} {}:{}:{}'.format(str(td.day).zfill(2), str(td.month).zfill(2), td.year, str(td.hour).zfill(2), str(td.minute).zfill(2), str(td.second).zfill(2)), (50, 85), ocv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, ocv.LINE_AA)
            cv_dat = ocv.putText(cv_dat, '{}-{}-{} {}:{}:{}'.format(str(td.day).zfill(2), str(td.month).zfill(2), td.year, str(td.hour).zfill(2), str(td.minute).zfill(2), str(td.second).zfill(2)), (50, 85), ocv.FONT_HERSHEY_SIMPLEX, 1, self.__color__, 1, ocv.LINE_AA)
            ocv.imwrite('/Frames/' + str(count).zfill(10) + '.jpg', cv_dat)
            ocv.imshow('Stream Classification : Livestream', cv_dat)
            if ocv.waitKey(25) & 0xFF == ord('q'):
                ocv.destroyAllWindows()
                break
            count += 1
        if os.path.exists(f'/{self.output_saving_addr}/Output_Video.mp4'):
            os.system(f'rm /{self.output_saving_addr}/Output_Video.mp4')
        os.system('\'' + self.fmpg_addr + '\' -framerate ' + str(self.__out_framerate__) + ' -i \'' + '/Frames/' + '%10d.jpg\' -c:v libx264 -r ' + str(self.__out_framerate__) + ' \'' + '/Output/Output_Video.mp4\'')
        os.system('rm /Frames/*')
        os.system(f'chmod 777 /{self.output_saving_addr}/*')

# %%
# Live Client Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stream Classification Live Inference Client.')
    parser.add_argument('-l', '--link', type = str, help = 'Live Data Link', required = True)
    parser.add_argument('-ip', '--server_ip', type = str, help = 'IP Address to GRPC Server => IP:Port', required = True)
    parser.add_argument('-sk', '--skip', type = int, help = 'Number of Frames to Skip in Live Inference', default = 0)
    parser.add_argument('-tz', '--time_zone', type = str, help = 'Current Time Zone', default = 'Asia/Karachi')
    parser.add_argument('-fmpg', '--ffmpeg', type = str, help ='Absolute Address to Standalone FFMPEG File', default = '/resources/ffmpeg')
    parser.add_argument('-ytdl', '--youtube_dl', type = str, help ='Absolute Address to Standalone Youtube-DL File', default = '/resources/youtube-dl')
    args = vars(parser.parse_args())
    liv = Live(args['link'], args['server_ip'], tzone = args['time_zone'], ytdl = args['youtube_dl'], ffmpeg = args['ffmpeg'])
    print(liv)
    liv(args['skip'])
    