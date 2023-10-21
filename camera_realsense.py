import numpy as np

from collections import namedtuple
from pyrealsense2 import pyrealsense2 as rs
from camera_utils.camera_capture import Capture

Stream = namedtuple('Stream', 'mode format')


class DepthFramePostProcessor:
    spatial_filter = rs.spatial_filter()
    spatial_filter.set_option(rs.option.filter_magnitude, 5)
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 1)
    spatial_filter.set_option(rs.option.filter_smooth_delta, 50)
    temporal_filter = rs.temporal_filter()
    depth_to_disparity = rs.disparity_transform()
    disparity_to_depth = rs.disparity_transform(False)
    hole_filling_filter = rs.hole_filling_filter()
    hole_filling_filter.set_option(rs.option.holes_fill, 2)
    
    @classmethod
    def process(cls, depth_frame):
        if not depth_frame:
            return
        
        frame = cls.depth_to_disparity.process(depth_frame)
        frame = cls.spatial_filter.process(frame)
        frame = cls.temporal_filter.process(frame)
        frame = cls.disparity_to_depth.process(frame)
        frame = cls.hole_filling_filter.process(frame)
        return frame


class RealSenseCamera(Capture):
    def __init__(self, width, height, fps=60, infrared=False, depth=False):
        super(RealSenseCamera, self).__init__(width, height)
        self.width = width
        self.height = height
        self.fps = fps
        self.infrared = infrared
        self.depth = depth
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = 0
        self.set_up()
    
    def set_up(self):
        streams = []
        self.handle_infrared(streams)
        self.handle_depth(streams)
        
        for stream in streams:
            self.config.enable_stream(stream.mode, self.width, self.height, stream.format, self.fps)
        
        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        
        if not self.depth:
            pass
            # depth_sensor.set_option(rs.option.emitter_enabled, 0)
        else:
            self.depth_scale = depth_sensor.get_depth_scale()
    
    def handle_infrared(self, streams):
        if self.infrared:
            streams.append(Stream(rs.stream.infrared, rs.format.y8))
            return
        
        streams.append(Stream(rs.stream.color, rs.format.bgr8))
    
    def handle_depth(self, streams):
        if self.depth:
            streams.append(Stream(rs.stream.depth, rs.format.z16))
    
    def read(self):
        super(RealSenseCamera, self).read()
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_infrared_frame() if self.infrared else aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame() if self.depth else None
        depth_frame = DepthFramePostProcessor.process(depth_frame)
        
        if not color_frame:
            return False, np.zeros((self.width, self.height, 3), np.uint8), depth_frame
        
        return True, np.asanyarray(color_frame.get_data()), depth_frame
    
    def release(self):
        super(RealSenseCamera, self).release()
        self.pipeline.stop()
