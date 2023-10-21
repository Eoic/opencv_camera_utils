import threading
from _queue import Empty
from queue import Queue

from camera_utils.camera_web import WebCamera


class BufferedWebCamera(WebCamera):
    def __init__(self, stream_uri):
        super(BufferedWebCamera, self).__init__(stream_uri)
        self.buffer = Queue()
        self.thread_stop = threading.Event()
        self.thread = threading.Thread(target=self.fill_buffer, args=(self.thread_stop,))
        self.thread.daemon = True
        self.thread.start()
    
    def fill_buffer(self, stop_event):
        while not stop_event.is_set():
            success, frame = self.capture.read()
            
            if not success:
                break
            if not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except Empty:
                    pass
        
            self.buffer.put(frame)
    
    def read(self):
        if self.buffer.not_empty:
            return True, self.buffer.get(), None
        
        return False, None, None
    
    def release(self):
        self.thread_stop.set()
        self.thread.join()
        super(BufferedWebCamera, self).release()
