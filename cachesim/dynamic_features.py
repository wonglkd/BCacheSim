#!/usr/bin/env python3
from collections import defaultdict, deque

import numpy as np


HOUR_IN_SECONDS = 3600


class DynamicFeatures:
    # revised bloom filter implementation: record number of occurrence
    # instead of "has occured" flag

    def __init__(self, hours, granularity='chunk', hr_unit=HOUR_IN_SECONDS):
        self.history = deque()  # queue broken down by hour slots.
        self.timestamps = deque()  # timestamps in seconds
        self.hours = hours
        self.last_access_time = {}
        self.granularity = granularity
        self.hr_unit = hr_unit

    def _key(self, key):
        if self.granularity.startswith('block') and type(key) != int and len(key) == 2:
            return key[0]
        return key

    def updateFeatures(self, key, ts, weight=1):
        key = self._key(key)
        # empty startup or past 1 hr: start a new set
        if len(self.history) == 0 or ts > self.timestamps[0] + self.hr_unit:
            self.history.appendleft(defaultdict(int))
            self.timestamps.appendleft(ts)
        # if too many sets, pop oldest (fifo)
        if len(self.history) > self.hours:
            self.history.pop()
            self.timestamps.pop()
        # update set with this key
        # if key not in self.history[0]:
        self.history[0][key] += weight
        # update time since last access
        self.last_access_time[key] = ts

    # only gets a single key's bloom-filter features
    def getFeature(self, key):
        key = self._key(key)
        feature = [
            s[key] for s in self.history
        ]  # return how many times the key has been accessed in each bucket

        feature.extend([0] * max(0, (self.hours - len(self.history))))
        return feature

    # only gets bloom-filter features
    def getFeatures(self, keys):
        features = []
        for key in keys:
            features.append(self.getFeature(key))
        return features

    def getLastAccessDistance(self, key, ts):
        key = self._key(key)
        if key not in self.last_access_time:
            return np.inf
        distance = ts - self.last_access_time[key]
        # don't record distance if that's greater than self.hours
        if distance > self.hours * self.hr_unit:
            return np.inf
        return distance

    def getLastAccessDistances(self, keys, ts):
        return [self.getLastAccessDistance(key, ts) for key in keys]

    def ready(self):
        return len(self.history) >= self.hours
