import torch
import datetime
import numpy as np


class Timer(object):

    def __init__(self):
        self.timestamps = []

    def __str__(self):
        return Timer.str_format(self.elapsed)

    def tap(self):
        now = datetime.datetime.now()
        self.timestamps.append(now)
        return now

    @property
    def elapsed(self):
        now = datetime.datetime.now()
        h, m, s, ms = Timer.get_time_diff(now, self.timestamps[0])
        return h, m, s, ms

    @property
    def delta(self):
        now = datetime.datetime.now()
        h, m, s, ms = Timer.get_time_diff(now, self.timestamps[-1])
        return h, m, s, ms

    @property
    def delta_str(self):
        return Timer.str_format(self.delta)

    @staticmethod
    def str_format(time):
        h, m, s, ms = time
        if h > 0:
            s = '%dh %dm %d.%ds' % time
        elif m > 0:
            s = '%dm %d.%ds' % time[1:]
        else:
            s = '%d.%ds' % time[2:]
        return s

    @staticmethod
    def get_time_diff(now, previous):
        dt = now - previous
        hours = dt.seconds // 3600
        minutes = (dt.seconds // 60) % 60
        seconds = dt.seconds % 60
        milliseconds = dt.microseconds // 1000
        return hours, minutes, seconds, milliseconds
