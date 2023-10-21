import argparse
from enum import Enum

from cvui import cvui, np
from cv2 import cv2
from camera_utils.calibrator import Calibrator
from camera_utils.camera_adapter import CameraAdapter
from camera_utils.camera_web_buffered import BufferedWebCamera
from camera_utils.utils.keys import Keys
from camera_utils.utils.runner import Runner


class Windows(Enum):
    MAIN = 0
    CLIPPED = 1

# python setup.py sdist bdist_wheel


class CameraRunner(Runner):
    def __init__(self):
        super(CameraRunner, self).__init__()
        self.arguments = self.parse_arguments()
        self.calibrator = Calibrator(Windows.MAIN.name)
        self.camera = BufferedWebCamera('rtsp://<username>:<password>@<ip address>')
        # self.camera = SnapshotCamera(self.arguments.source)
        # self.camera = RealSenseCamera(640, 480, fps=30)
        self.camera_adapter = CameraAdapter(self.camera)
        self.calibrator.attach(self.camera_adapter)
        self.frame = None
        self.depth_frame = None
        self.create_windows()
    
    @classmethod
    def create_windows(cls):
        cvui.init(Windows.MAIN.name)

    def parse_arguments(self):
        super(CameraRunner, self).parse_arguments()
        parser = argparse.ArgumentParser()
        parser.add_argument('--source', help='Path to the video stream or file.', default=0)
        parser.add_argument('--headless', help='Hide debug windows.', action='store_const', const=True)
        parser.add_argument('--still', help='Processing still frame.', action='store_const', const=True)
        parser.add_argument('--host', help='Shapes data broadcast host.', action='store_const', const='localhost', default='localhost')
        parser.add_argument('--port', help='Shapes data broadcast port.', action='store_const', const=8000, default=8000)
        return parser.parse_args()
    
    def handle_input(self):
        super(CameraRunner, self).handle_input()
        
        if self.key == Keys.ESC.value:
            self.is_running = False
        elif self.key == Keys.C.value:
            self.calibrator.start_editing(self.frame, self.camera_adapter)
    
    def set_up(self):
        self.camera.record(300, 25, (1920, 1080))
    
    def update(self, delta=None):
        super(CameraRunner, self).update()
        _, self.frame, _ = self.camera_adapter.read()
        clipped = self.calibrator.get_clipped(self.frame, copy=True)
        transform = self.calibrator.transform_perspective(self.frame)
        masked_frame = self.calibrator.get_masked(transform)
        cv2.imshow('Masked frame', clipped)
        self.calibrator.update(self.frame, True)
        self.camera_adapter.update(self.frame)
        
        if isinstance(self.frame, np.ndarray):
            cvui.imshow(Windows.MAIN.name, self.frame)


if __name__ == '__main__':
    CameraRunner().raw_run()
