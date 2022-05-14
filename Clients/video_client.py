#!/usr/bin/env python3

"""
STREAM CLASSIFICATION LOCAL INFERENCE CLIENT
============================================

The following program is used to perform local data inference on the subject data
"""

# %%
# Importing Libraries
from inference_client import *
import os
import argparse
import cv2 as ocv

# %%
# Main Local Video Client Inference Class
class Video:
    def __init__(self, addr, server_ip, ffmpeg='/resources/ffmpeg'):
        """
        This method is used to initialize Stream Classification local video inference client

        Method Input
        =============
        addr : Local address to data
        server_ip : Server IP at which GRPC server is running
                            Format : "IP:Port"
                            Example : '0.0.0.0:1234'
        ffmpeg : Absolute address of FFMPEG standalone file

        Method Output
        ==============
        None
        """
        self.vid_addr = addr
        self.fmpg_addr = ffmpeg
        self.sc_server_ip = server_ip
        self.__color__ = (0, 0, 255)
        self.__out_framerate__ = 30
        self.output_saving_addr = '/Output'
        self.__vid_obj__ = ocv.VideoCapture(self.vid_addr)
        self.FPS = int(self.__vid_obj__.get(ocv.CAP_PROP_FPS))
        self.height = int(self.__vid_obj__.get(ocv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.__vid_obj__.get(ocv.CAP_PROP_FRAME_WIDTH))
        self.__total_frames__ = int(self.__vid_obj__.get(ocv.CAP_PROP_FRAME_COUNT))
        self.stream_inference = sc_client(self.sc_server_ip)
        if not os.path.exists('/Frames'):
            os.mkdir('/Frames')
    
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
        ================================================
        | Stream Classification Local Inference Client |
        ================================================
        """)
        print(f'Stream Classification Server IP Address: {self.sc_server_ip}')
        print(f'Client ID: {self.stream_inference.client_name}')
        print(f'Video FPS: {self.FPS}')
        print(f'Video Height: {self.height}')
        print(f'Video Width: {self.width}')
        print(f'Total Video Frames: {self.__total_frames__}')
        print(f'Output Frame Rate: {self.__out_framerate__}')
        print(f'Output Data Saving Directory: {self.output_saving_addr}')
        print('\n---------------------------------------------')
        print('>>>>> Press q to End Stream')
        return '\n'
    
    def __call__(self, skip = 0):
        """
        This method is used to handle & perform inference on local data

        Method Input
        =============
        skip : Number of frames to skip in local video inference ( default : 0 )

        Method Output
        ==============
        None
        """
        count, skip_count, ret = 0, skip, True
        csv_str = 'Frame #,Stream Classification\n'
        while ret:
            ret, cv_dat = self.__vid_obj__.read()
            if not ret:
                break
            if skip_count < skip:
                skip_count += 1
            else:
                skip_count = 0
                pil_dat = Image.fromarray(ocv.cvtColor(cv_dat, ocv.COLOR_BGR2RGB))
                res = self.stream_inference([pil_dat])
                csv_str += f'{count},{res[0][0]}\n'
                with open(f'/{self.output_saving_addr}/Output_Metadata.csv', 'w') as file1:
                    file1.write(csv_str)
            cv_dat = ocv.putText(cv_dat, res[0][0], (50, 50), ocv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, ocv.LINE_AA)
            cv_dat = ocv.putText(cv_dat, res[0][0], (50, 50), ocv.FONT_HERSHEY_SIMPLEX, 1, self.__color__, 1, ocv.LINE_AA)
            ocv.imwrite('/Frames/' + str(count).zfill(10) + '.jpg', cv_dat)
            ocv.imshow('Stream Classification : Local Video', cv_dat)
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
# Local Video Client Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stream Classification Local Video Inference Client.')
    parser.add_argument('-ip', '--server_ip', type = str, help = 'IP Address to GRPC Server => IP:Port', required = True)
    parser.add_argument('-l', '--link', type = str, help = 'Local Video Address', default = '/video.mp4' )
    parser.add_argument('-sk', '--skip', type = int, help = 'Number of Frames to Skip in Local Video Inference', default = 0)
    parser.add_argument('-fmpg', '--ffmpeg', type = str, help ='Absolute Address to Standalone FFMPEG File', default = '/resources/ffmpeg')
    args = vars(parser.parse_args())
    vid = Video(args['link'], args['server_ip'], ffmpeg = args['ffmpeg'])
    print(vid)
    vid(args['skip'])
