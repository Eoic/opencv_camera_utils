from enum import Enum


class ObservationEvent(Enum):
    SAVE_BACKGROUND = 0
    CALIBRATION_DONE = 1


class Observer:
    def handle_event(self, event):
        pass
