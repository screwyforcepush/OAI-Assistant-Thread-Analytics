class ThreadObject:
    def __init__(self, thread_id, messages, started_time, finished_time, model):
        self.thread_id = thread_id
        self.messages = messages
        self.started_time = started_time
        self.finished_time = finished_time
        self.model = model

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            thread_id=data['thread_id'],
            messages=data['messages'],
            started_time=data['started_time'],
            finished_time=data['finished_time'],
            model=data['model']
        )
        # Set any additional attributes from the dictionary
        for key, value in data.items():
            setattr(obj, key, value)
        return obj
