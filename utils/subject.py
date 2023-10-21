class Subject:
    def __init__(self):
        self.observers = []
    
    def attach(self, observer):
        self.observers.append(observer)
    
    def notify_all(self, event):
        for observer in self.observers:
            observer.handle_event(event)
