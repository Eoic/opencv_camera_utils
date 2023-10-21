from cv2 import cv2

from camera_utils.camera_capture import Capture


class SnapshotCamera(Capture):
    def __init__(self, stream_uri):
        self.frame = cv2.imread(stream_uri)
        height, width = self.frame.shape[:2]
        self.width = width
        self.height = height
        super(SnapshotCamera, self).__init__(self.width, self.height)
    
    def read(self):
        super(SnapshotCamera, self).read()
        return True, self.frame, None
    
    def release(self):
        super(SnapshotCamera, self).release()
