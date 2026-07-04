from collections import deque


class PathState:
    def __init__(self, history_frames=3):
        self.history_frames = history_frames
        self.members = {}
        self.missed = {}
        self.last = {}
        self.records = {}

    def observe(self, path_id, candidate, record=None):
        frames = self.members.setdefault(path_id, deque(maxlen=self.history_frames))
        frames.append(set(candidate.members))
        self.last[path_id] = candidate
        if record is not None:
            self.records[path_id] = record
        self.missed[path_id] = 0

    def miss(self, path_id):
        value = self.missed.get(path_id, 0) + 1
        self.missed[path_id] = value
        return value

    def clear(self, path_id):
        self.members.pop(path_id, None)
        self.missed.pop(path_id, None)
        self.last.pop(path_id, None)
        self.records.pop(path_id, None)
