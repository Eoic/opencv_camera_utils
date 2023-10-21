import time
from cv2 import cv2


class Runner:
    def __init__(self, framerate=60):
        self.wait_time = 1e9 / framerate
        self.catch_up_time = 0
        self.is_running = False
        self.key = None

    def set_up(self):
        """ Called before update loop is started """
        pass

    def teardown(self):
        """ Called once update loop is done. """
        pass

    def update(self, delta=None):
        """ Other processing """
        pass

    def handle_input(self):
        """ Input processing """
        self.key = cv2.waitKey(1)
        pass

    def run(self):
        """ Run at defined time step (fps). """
        self.is_running = True
        last_loop_time = time.time_ns()
        self.set_up()

        while self.is_running:
            now = time.time_ns()
            tick_length = now - last_loop_time
            last_loop_time = now
            delta = tick_length / self.wait_time
            self.handle_input()
            self.update(delta)
            yield_time = ((last_loop_time - time.time_ns() + self.wait_time) / 1e9) + self.catch_up_time
            self.catch_up_time = 0

            if yield_time > 0:
                time.sleep(yield_time)
            else:
                self.catch_up_time += yield_time

        self.teardown()
    
    def raw_run(self):
        """ Process without fixed time step. """
        self.is_running = True
        self.set_up()

        while self.is_running:
            self.handle_input()
            self.update()

        self.teardown()
        
    def parse_arguments(self):
        pass
