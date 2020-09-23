# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time
import math


class CentroidTracker:
    def angle_between(self, a, b):
        # print(a,b)
        y = b[1] - a[1]
        x = b[0] - a[0]
        # print(y, x)
        # print(self.x_th, self.y_th)
        if x > self.x_th:
            return "in"
        elif x < -self.x_th:
            return "out"
        else:
            return "standing"

    def __init__(self, maxDisappeared=10, maxDistance=150, x_th = 0, y_th = 0):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.x_th = x_th
        self.y_th = y_th

    def register(self, centroid):
        # print("ok")
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = (centroid, 0, (0, 0))
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [w[0] for w in self.objects.values()]
            # print(objectCentroids, "----\n", inputCentroids)
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                # print("checking: ", inputCentroids[col])
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                x, y = self.objects[objectID][0]
                # print(objectID)
                # print("updated", inputCentroids[col])
                self.objects[objectID] = (inputCentroids[col], self.angle_between((x, y), inputCentroids[col]), (x, y))
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            # print("used cols: ", usedCols)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # print("unused cols: ", unusedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # else:
            for col in unusedCols:
                self.register(inputCentroids[col])
        return self.objects
