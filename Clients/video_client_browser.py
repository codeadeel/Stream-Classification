#!/usr/bin/env python3

"""
STREAM CLASSIFICATION LOCAL INFERENCE CLIENT IN WEB BROWSER
===========================================================

The following program is used to perform local data inference on the subject data & display results in web browser
"""

# %%
# Importing Libraries
from inference_client import *
import time
import argparse
import cv2 as ocv
from flask import Flask, render_template, Response

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
        self.csv_str = 'Frame #,Stream Classification,Brand Recognition\n'
    
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
        ==============================================================
        | Stream Classification Browser Based Local Inference Client |
        ==============================================================
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
            time.sleep(1/self.FPS)
            self.count += 1
            dummy, cv_dat = ocv.imencode('.jpg', cv_dat)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv_dat.tobytes())

# %%
# Browser Based Local Video Client Execution
app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('display.html')
@app.route('/data_stream')
def streamer():
    return Response(vid(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stream Classification Browser Based Local Video Inference Client.')
    parser.add_argument('-ip', '--server_ip', type = str, help = 'IP Address to GRPC Server => IP:Port', required = True)
    parser.add_argument('-cip', '--client_ip', type = str, help = 'IP Address for Client => IP:Port', default = '0.0.0.0:80')
    parser.add_argument('-l', '--link', type = str, help = 'Local Video Address', default = '/video.mp4' )
    parser.add_argument('-sk', '--skip', type = int, help = 'Number of Frames to Skip in Local Video Inference', default = 0)
    args = vars(parser.parse_args())
    vid = Video(args['link'], args['server_ip'], skip = args['skip'])
    print(vid)
    cip_ip = args['client_ip'].split(':')[0]
    cip_port = int(args['client_ip'].split(':')[1])
    app.run(host = cip_ip, port = cip_port)
        