import glob
import traceback
from os import path, makedirs
import cvui
import numpy as np
from cv2 import cv2
from camera_utils.utils.colors import Color
from camera_utils.utils.observer import ObservationEvent
from camera_utils.utils.subject import Subject


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    @property
    def tuple(self):
        return int(self.x), int(self.y)
    
    @property
    def array(self):
        return [self.x, self.y]
    
    def square(self, size):
        half = size / 2
        return (int(self.x - half), int(self.y - half)), \
               (int(self.x + half), int(self.y + half))
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __repr__(self):
        return str(self)


class Rect:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = int(x)
        self.y = int(y)
        self.w = int(x + w)
        self.h = int(y + h)
    
    def expand(self):
        return self.x, self.y, self.w, self.h
    
    def __str__(self):
        return f"({self.x}, {self.y}) ({self.w}, {self.h})"


class Calibrator(Subject):
    def __init__(self, window_name):
        super(Calibrator, self).__init__()
        self.settings_path = 'data'
        self.settings_name = 'calibration.txt'
        self.background_name = 'background.png'
        self.clip_points = []
        self.frame = None
        self.clip_frame = {}
        self.edit_mode = False
        self.limit_visibility = False
        self.window_name = window_name
        
        # Clip area
        self.clip_dragging = [False, False, False, False]
        self.has_clip_area = False
        self.clip_dirty = False
        self.clip_drag_handle_size = 20
        self.clip_drag_size = 300
        self.clip_rect = Rect()
        
        # Visibility area (ROI mask)
        self.visibility_bound_points = []
        self.visibility_mask_vertices = []
        self.visibility_mask = None
        self.visibility_rect = Rect()
        
        # Perspective
        self.perspective_matrix = None
        self.__load_points()
    
    @property
    def width(self):
        return self.clip_rect.w - self.clip_rect.x
    
    @property
    def height(self):
        return self.clip_rect.h - self.clip_rect.y
    
    @property
    def area(self):
        """ Return clipped area size. """
        return self.width * self.height
    
    @property
    def clip_size(self):
        """ Returns size of the clipped rectangle. """
        return [self.width, self.height]
    
    def start_editing(self, frame, camera=None):
        """ Enter to edit mode. """
        if self.edit_mode:
            self.edit_mode = False
            
            if camera is not None:
                camera.is_paused = False
            
            return
        
        if len(self.clip_points) == 4:
            self.clip_points = []
            self.visibility_bound_points = []
            self.has_clip_area = False
        
        cv2.setMouseCallback(self.window_name, self.set_point_callback)
        
        if camera is not None:
            camera.is_paused = True
        
        self.edit_mode = True
        self.frame = frame
    
    def stop_editing(self):
        """ Exit from edit mode. """
        cv2.setMouseCallback(self.window_name, lambda *args: None)
        self.edit_mode = False
        
        if len(self.clip_points) == 4:
            self.has_clip_area = True
            self.calculate_ranges()
            self.compute_visibility_mask()
            self.save_points()
        
        cvui.init(self.window_name)
    
    def compute_visibility_mask(self):
        """ Create ROI where image will be visible. """
        vis_x, vis_y, vis_width, vis_height = self.visibility_rect.expand()
        clip_x, clip_y, clip_width, clip_height = self.clip_rect.expand()
        frame_mask = np.zeros((vis_height - vis_y, vis_width - vis_x), dtype=np.uint8)
        mask_vertices = np.array([[[clip_x - vis_x, clip_y - vis_y],
                                   [clip_x - vis_x, clip_height - vis_y],
                                   [clip_width - vis_x, clip_height - vis_y],
                                   [clip_width - vis_x, clip_y - vis_y]]], dtype=np.int32)
        self.visibility_mask = cv2.fillPoly(frame_mask, mask_vertices, 255)
    
    @classmethod
    def calculate(cls, points):
        """ Calculate clipping rectangle bounds from clip points. """
        if len(points) > 2:
            min_x = points[0].x
            max_x = points[0].x
            min_y = points[0].y
            max_y = points[0].y
            
            for point in points:
                min_x = min(min_x, point.x)
                max_x = max(max_x, point.x)
                min_y = min(min_y, point.y)
                max_y = max(max_y, point.y)
            
            return min_x, min_y, max_x - min_x, max_y - min_y
        return 0, 0, 0, 0
    
    def calculate_ranges(self):
        """ Create clipping rectangle from clip points. """
        x_clip, y_clip, w_clip, h_clip = self.calculate(self.clip_points)
        x_vis, y_vis, w_vis, h_vis = self.calculate(self.visibility_bound_points)
        self.clip_rect = Rect(x_clip, y_clip, w_clip, h_clip)
        self.visibility_rect = Rect(x_vis, y_vis, w_vis, h_vis)
        self.compute_perspective_matrix()
        print('[CALIBRATOR] Recalculating calibration area.')
    
    def compute_perspective_matrix(self):
        """ Calculate perspective matrix from arranged visibility points for wrapping. """
        if len(self.visibility_bound_points) != 4:
            return
        
        vis_x, vis_y, vis_width, vis_height = self.visibility_rect.expand()
        sorted_by_y = sorted(self.visibility_bound_points, key=lambda point: point.y)
        upper_half = sorted(sorted_by_y[:2], key=lambda point: point.x)
        lower_half = sorted(sorted_by_y[2:], key=lambda point: point.x, reverse=True)
        sorted_points = upper_half + lower_half
        source_points = np.float32([[[point.x, point.y] for point in sorted_points]])
        destination_points = np.float32([[0, 0],
                                         [vis_width - vis_x, 0],
                                         [vis_width - vis_x, vis_height - vis_y],
                                         [0, vis_height - vis_y]])
        self.perspective_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    
    def transform_perspective(self, frame):
        """ Transform perspective of the given frame according to visibility points. """
        vis_x, vis_y, vis_width, vis_height = self.visibility_rect.expand()
        return cv2.warpPerspective(frame, self.perspective_matrix, (vis_width - vis_x, vis_height - vis_y))
    
    def set_point_callback(self, event, x, y, flags, param):
        """ Fired upon double click on the image. """
        if not self.edit_mode:
            return
        
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if len(self.clip_points) < 4:
                self.clip_points.append(Point(x, y))
                self.visibility_bound_points.append(Point(x, y))
            
            if len(self.clip_points) == 4:
                self.stop_editing()
                self.notify_all(ObservationEvent.CALIBRATION_DONE)
    
    def get_clipped(self, frame, copy=False):
        """ Return partial frame clipped from defined rectangle. If no clipping points were set, return full frame. """
        clipped_frame = frame
        
        try:
            clipped_frame = frame[self.clip_rect.y:self.clip_rect.h, self.clip_rect.x:self.clip_rect.w]
        except (Exception, IndexError) as _:
            traceback.print_exc()
        finally:
            if isinstance(clipped_frame, np.ndarray) and clipped_frame.any():
                return clipped_frame.copy() if copy else clipped_frame
            
            return frame.copy() if copy else frame
    
    def get_masked(self, frame):
        """ Return masked image if mask area is defined. """
        if isinstance(self.visibility_mask, np.ndarray):
            return cv2.copyTo(frame, mask=self.visibility_mask)
    
    def update(self, frame, show_clip_area=False):
        """ Display clipping points in edit mode. """
        cvui.context(self.window_name)
        
        if self.edit_mode:
            h = frame.shape[0]
            cv2.putText(frame, "Calibrating mode", (10, h - 20), cv2.FONT_HERSHEY_PLAIN, 2, Color.GREEN.value, 1)
            
            for point in self.clip_points:
                cv2.circle(frame, (point.x, point.y), 5, Color.GREEN.value, 2)
        
        if show_clip_area:
            clip_point_count = len(self.clip_points)
            
            if clip_point_count != 4:
                return
            
            self.show_markers(frame)
            cvui.update()
    
    @classmethod
    def draw_bounds(cls, frame, points, line_color):
        """ Draw rectangle from given points. """
        if len(points) < 2:
            return
        
        cv2.line(frame, points[0].tuple, points[-1].tuple, line_color, 2)
        
        for i in range(len(points) - 1):
            cv2.line(frame, points[i].tuple, points[i + 1].tuple, line_color, 3)
    
    @classmethod
    def clamp_mouse_position(cls, point, min_x, min_y, max_x, max_y):
        """ Limit mouse coordinates to given bounds. """
        mouse_x = cvui.mouse().x
        mouse_y = cvui.mouse().y
        
        if mouse_x < min_x or mouse_x > max_x:
            mouse_x = point.x
        
        if mouse_y < min_y or mouse_y > max_y:
            mouse_y = point.y
        
        return mouse_x, mouse_y
    
    def show_markers(self, frame):
        """ Display clip area handles. """
        frame_height, frame_width = frame.shape[:2]
        point_count = len(self.clip_points)
        
        # Actual bounding box.
        self.draw_bounds(frame, self.clip_points, Color.GREEN.value)
        self.draw_bounds(frame, self.visibility_bound_points, Color.LIGHT_BLUE.value)
        cv2.rectangle(frame, (self.clip_rect.x, self.clip_rect.y), (self.clip_rect.w, self.clip_rect.h),
                      Color.RED.value, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (self.visibility_rect.x, self.visibility_rect.y),
                      (self.visibility_rect.w, self.visibility_rect.h), Color.LIGHT_RED.value, 2, cv2.LINE_AA)
        
        # Clip points
        for i in range(point_count):
            x, y = self.clip_points[i].tuple
            drag_size = self.clip_drag_size if self.clip_dragging[i] else self.clip_drag_handle_size
            square_left, square_right = self.clip_points[i].square(self.clip_drag_handle_size)
            upper_left_x = int(x - drag_size / 2)
            upper_left_y = int(y - drag_size / 2)
            status = cvui.iarea(upper_left_x, upper_left_y, drag_size, drag_size)
            
            if status == cvui.OVER:
                self.clip_dragging[i] = False
                cv2.rectangle(frame, square_left, square_right, Color.RED.value, 2, cv2.LINE_AA)
            elif status == cvui.DOWN:
                cv2.rectangle(frame, square_left, square_right, Color.YELLOW.value, 2, cv2.LINE_AA)
                
                if self.visibility_rect.w > 0 and self.visibility_rect.h > 0 and self.limit_visibility:
                    mouse_x, mouse_y = self.clamp_mouse_position(self.clip_points[i], *self.visibility_rect.expand())
                else:
                    mouse_x, mouse_y = self.clamp_mouse_position(self.clip_points[i], 0, 0, frame_width, frame_height)
                
                for j in range(len(self.clip_points)):
                    self.clip_dragging[j] = False
                
                self.clip_dragging[i] = True
                
                if square_left[0] < x < square_right[0] and square_left[1] < y < square_right[1]:
                    self.clip_points[i] = Point(mouse_x, mouse_y)
                    self.calculate_ranges()
                    self.compute_visibility_mask()
                    self.clip_dirty = True
            
            elif status == cvui.OUT:
                self.clip_dragging[i] = False
                cv2.rectangle(frame, square_left, square_right, Color.BLUE.value, 2, cv2.LINE_AA)
                
                if self.clip_dirty:
                    self.save_points()
    
    def save_points(self):
        """ Save defined clipping rectangle points to data file. """
        if not path.exists(self.settings_path):
            makedirs(self.settings_path)
        
        with open('/'.join([self.settings_path, self.settings_name]), 'w+') as file:
            for points in [self.clip_points, self.visibility_bound_points]:
                for point in points:
                    file.writelines([f"{point.x} {point.y}\n"])
    
    def __load_points(self):
        """ Load clipping rectangle points from data file. """
        settings_dir = '/'.join([self.settings_path, self.settings_name])
        
        if path.exists(settings_dir):
            with open(settings_dir, 'r') as file:
                self.clip_points = self.__read_points(file)
                self.visibility_bound_points = self.__read_points(file)
        
        if len(self.clip_points) == 4:
            self.has_clip_area = True
            self.calculate_ranges()
            self.compute_visibility_mask()
    
    def normalize_clipped(self, x, y):
        """ Normalize given point to coordinates in range (0, 1) according to clipping rectangle size. """
        return x / (self.clip_rect.w - self.clip_rect.x), y / (self.clip_rect.h - self.clip_rect.y)
    
    @classmethod
    def normalize(cls, point, width, height):
        """ Normalize given point to coordinates in range (0, 1) according custom rectangle size. """
        return point[0] / width, point[1] / height
    
    @classmethod
    def __read_points(cls, file):
        """ Load clipping points from data file. """
        read_points = []
        
        for i in range(4):
            point = file.readline().strip('\n').split(' ')
            
            if len(point) == 2:
                read_points.append(Point(int(point[0]), int(point[1])))
            else:
                read_points.append(Point(0, 0))
        
        return read_points
