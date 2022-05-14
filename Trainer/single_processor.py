#!/usr/bin/env python3

"""
STREAM CLASSIFICATION DATA PRE-PROCESSOR FOR SINGLE VIDEO
=========================================================

The following program is used to prepare data for model training
"""

# %%
# Importing Libraries
import os
import argparse
import tqdm
import numpy as np
import cv2 as ocv

# %%
# Main Data Processing Class
class Video:
    def __init__(self, vid_addr, ext_addr, ext_label, skip = 0, dims = (0,0), quality = 50):
        """
        This method is used to initialize video processing class

        Method Input
        =============
        vid_addr : Absolute address of video file
        ext_addr : Absolute directory address to save processed data
        ext_label : Label for the current video extraction
        skip : Number of frames to skip in video processing ( Default : 0)
        dims : Target frame dimensions ( Default : ( width, height ) :: ( 0, 0 ) )
        quality : Quality of storing image ( 1 - 100 ) (Default : 50)
        
        Method Output
        ==============
        None
        """
        self.video_address = vid_addr
        self.extraction_address = ext_addr
        self.extraction_label = ext_label
        self.skip = skip
        self.extraction_quality = quality
        self.file_name = self.video_address.split('/')[-1].split('.')[0]
        self.__video__ = ocv.VideoCapture(self.video_address)
        self.FPS = int(self.__video__.get(ocv.CAP_PROP_FPS))
        self.height = int(self.__video__.get(ocv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.__video__.get(ocv.CAP_PROP_FRAME_WIDTH))
        self.__total_frames__ = int(self.__video__.get(ocv.CAP_PROP_FRAME_COUNT))
        try:
            self.target_dimensions = dims
            if self.target_dimensions == (0,0):
                raise Exception('Null Value')
        except:
            self.target_dimensions = (self.height, self.width)
    
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
        =========================================
        | Stream Classification Data Processing |
        =========================================
        """)
        print(f'Video Address: {self.video_address}')
        print(f'Data Extraction Address: {self.extraction_address}')
        print(f'Data Extraction Label: {self.extraction_label}')
        print(f'Video File Name: {self.file_name}')
        print(f'Frame Skipping: {self.skip}')
        print(f'Frame Extraction Quality: {self.extraction_quality}')
        print(f'Frame Extraction Resolution: {self.target_dimensions[1]} x {self.target_dimensions[0]}')
        print(f'Video Base Resolution: {self.width} x {self.height}')
        print(f'Video Frames Per Second: {self.FPS}')
        print(f'Total Number of Frames: {self.__total_frames__}')
        print('\n---------------------------------------------')
        print()
        return '\n'
    
    def __call__(self):
        """
        This method is used to process frames and save them in target directory

        Method Input
        =============
        None
        
        Method Output
        ==============
        None
        """
        with tqdm.tqdm(total=self.__total_frames__, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True) as bar:
            count, skip_count, ret = 0, self.skip, True
            while ret:
                ret, data = self.__video__.read()
                if not ret:
                    break
                if skip_count < self.skip:
                    skip_count += 1
                else:
                    skip_count = 0
                    if (self.height, self.width) != self.target_dimensions:
                        data = ocv.resize(data, self.target_dimensions)
                    ocv.imwrite(f'{self.extraction_address}/{self.extraction_label}_{self.file_name}_{count}.jpg', data, [int(ocv.IMWRITE_JPEG_QUALITY), self.extraction_quality])
                bar.set_description('Extracting: {:<15} | Progress'.format(self.extraction_label))
                bar.update(1)
                count += 1

# %%
# Processing Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stream Classification Data Processor.')
    parser.add_argument('-l', '--label', type = str, help = 'Label for Current Video Extraction', required = True)
    parser.add_argument('-v', '--video', type = str, help = 'Video to Extract Frames From', default = '/Video.mp4')
    parser.add_argument('-t', '--target', type = str, help = 'Target Directory to Extract Frames', default = '/Output')
    parser.add_argument('-sk', '--skip', type = int, help = 'Frames to Skip During Extraction', default = 0)
    parser.add_argument('-q', '--quality', type = int, help = 'Quality of Storing Image ( 1 - 100 )', default = 50)
    parser.add_argument('-tw', '--target_width', type = int, help = 'Target Frame Width', default = 0)
    parser.add_argument('-th', '--target_height', type = int, help = 'Target Frame Height', default = 0)
    args = vars(parser.parse_args())
    vid = Video(vid_addr = args['video'], ext_addr = args['target'], ext_label = args['label'], skip = args['skip'], dims = (args['target_width'], args['target_height']), quality = args['quality'])
    print(vid)
    vid()
