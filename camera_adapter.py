from os import path

import numpy as np
from cv2 import cv2

from camera_utils.utils.colors import Color
from camera_utils.utils.observer import Observer, ObservationEvent


class CameraAdapter(Observer):
    def __init__(self, camera, is_still=False, calibrator=None):
        self.camera = camera
        self.frame = None
        self.depth_frame = None
        self.is_paused = False
        self.is_still = is_still
        self.settings_path = '../data'
        self.background_name = 'background.png'
        self.background_img = None
        self.width = self.camera.width
        self.height = self.camera.height
        self.calibrator = calibrator
        self.load_background()
    
    def read(self):
        """ Read and return next frame. """
        if self.is_still:
            success, frame, self.depth_frame = self.camera.read()
            
            if success:
                self.frame = frame
            
            return success, self.frame.copy(), None
        
        if not self.is_paused:
            _, self.frame, self.depth_frame = self.camera.read()
        
        if not isinstance(self.frame, np.ndarray):
            # self.is_paused = True
            return False, np.zeros((self.camera.width, self.camera.height, 3), np.uint8), None
        
        return True, self.frame.copy(), self.depth_frame
    
    def handle_event(self, event):
        """ Handle events received from subject. """
        if event == ObservationEvent.SAVE_BACKGROUND:
            self.save_background(self.frame)
        elif event == ObservationEvent.CALIBRATION_DONE:
            self.is_paused = False
    
    def load_background(self):
        """ Load saved background image from file (if any). """
        background_path = path.join(self.settings_path, self.background_name)
        
        if not path.exists(background_path):
            return
        
        self.background_img = cv2.imread(background_path)
    
    def save_background(self, frame):
        """ Save current frame. """
        background_path = path.join(self.settings_path, self.background_name)
        frame = self.calibrator.get_clipped(frame, copy=True) if self.calibrator else frame
        cv2.imwrite(background_path, frame)

    def update(self, frame):
        """ Display camera related info on the frame. """
        if self.is_paused:
            h, w = frame.shape[:2]
            cv2.putText(frame, "Paused", (w - 150, h - 20), cv2.FONT_HERSHEY_PLAIN, 2, Color.RED.value, 1)
    
    @property
    def background(self):
        """ Returns saved background image. """
        if not isinstance(self.background_img, np.ndarray):
            return self.frame
        
        return self.background_img
    
    def release(self):
        """ Stop capture. """
        self.camera.release()
