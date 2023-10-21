import datetime
import os
from time import sleep

from cv2 import cv2
from camera_utils.camera_capture import Capture


class WebCamera(Capture):
    def __init__(self, stream_uri):
        self.capture = cv2.VideoCapture(stream_uri) if stream_uri is not None else None
        self.width = int(self.capture.get(3))
        self.height = int(self.capture.get(4))
        super(WebCamera, self).__init__(self.width, self.height)
    
    def read(self):
        super(WebCamera, self).read()
        success, frame = self.capture.read()
        return success, frame, None
    
    def record(self, length, fps, size, output=None):
        default_dir = 'data/recordings'
        codec = cv2.VideoWriter_fourcc(*'XVID')
        filename = '{}/{}.avi'.format(default_dir, str(datetime.datetime.now()))
        video_writer = cv2.VideoWriter(filename, codec, fps, size)
        
        if length <= 0:
            raise ValueError('Invalid video recording length.')
        
        if not output:
            if not os.path.exists(default_dir):
                os.makedirs(default_dir)
        
        for i in range(length):
            _, frame, _ = self.read()
            video_writer.write(frame)
            sleep(fps / 1000)
    
    def release(self):
        super(WebCamera, self).release()
        self.capture.release()
