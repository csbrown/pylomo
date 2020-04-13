import time
import numpy as np
import collections

def fixed_width_formatter(number, n):
    places = np.log10(np.abs(number))
    if abs(places) == np.inf:
        places = 0
    highest_place = -int(places)
    if 1 <= highest_place < 3:
        rounded = np.round(number, n - highest_place - 1)
    elif highest_place >= 3:
        rounded = np.round(number, highest_place + n - 5)
    elif -n < highest_place < 1:
        rounded = np.round(number, n + highest_place - 2)
    else:
        rounded = np.round(number, highest_place + n - 6)
    return "{{:{}.{}g}}".format(n,n).format(rounded)

class TimeLogger(object):
    def __init__(self, logger=None, how_often=1, total=None, tag="", running_avg_length=None):
        self.logger = logger
        self.how_often = how_often
        self.total = total
        self.tag = tag
        self.times = collections.deque('',running_avg_length)
    def _is_print_time(self):
        return not self.i%self.how_often
    def __enter__(self):
        self.i = 0
        self.start = time.time()
    def __exit__(self, type, value, traceback):
        self.times.append(time.time() - self.start)
        if self._is_print_time():
            log_string = self.tag + " ... " if self.tag else ""#"avg: {}(m) ... left: {}(h) ... pct: {}"
            average_time = sum(self.times)/len(self.times)
            log_string += "avg: {}(m)".format(fixed_width_formatter(average_time/60, 8))
            if self.total is not None:
                estimated_time_remaining = (self.total-self.i)*average_time
                log_string += " ... left: {}(h)".format(fixed_width_formatter(estimated_time_remaining/60/60, 8))
                percent_complete = self.i/self.total
                log_string += " ... pct: {}".format(fixed_width_formatter(percent_complete, 8))
            self.logger.info(log_string)
        self.i += 1
