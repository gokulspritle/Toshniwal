import time
import math
import networkx as nx
from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict


class Object:
    # rect = l,t,b,r
    def __init__(self, pos):
        self.pos = pos
        self.last_seen = time.time()
        self.first_seen = time.time()

    def update(self, pos):
        self.pos = pos
        self.last_seen = time.time()


class Tracker:
    def __init__(self, drop_unseen_thresh=20):
        self.objects = OrderedDict()
        self.ids = 0
        self.drop_unseen_thresh = drop_unseen_thresh  # if unseen for 10seconds it will de deleted

    @staticmethod
    def __distance_between(rect1, rect2):
        x1, y1, x2, y2 = rect1[:4]
        a1, b1, a2, b2 = rect2[:4]
        c1 = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
        c2 = (a1 + (a2 - a1) / 2, b1 + (b2 - b1) / 2)
        dist = math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
        return dist

    @staticmethod
    def __centroid(rect):
        x1, y1, x2, y2 = rect
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def __dist_inverse(score_matrix):
        inv_scr = []
        lit = 10000
        for i in range(len(score_matrix)):
            temp = []
            for j in range(len(score_matrix[i])):
                scr = lit - score_matrix[i][j]
                if scr < 9980:
                    scr = -1
                temp.append(scr)
            inv_scr.append(temp)
        return inv_scr

    def update(self, new_positions):
        cent_rect = {self.__centroid(rect): tuple(rect) for rect in new_positions}
        now = time.time()
        # initial add
        if len(new_positions) == 0:
            return self.objects, cent_rect

        if len(self.objects) == 0:
            for rect in new_positions:
                self.objects[self.ids] = Object(self.__centroid(rect))
                self.ids += 1
            return self.objects, cent_rect
        # assign centroids with nearest distance for tracking
        score_matrix = dist.cdist(np.array([w.pos for w in self.objects.values()]),
                                  np.array([self.__centroid(rect) for rect in new_positions]))
        # print("-----", score_matrix)
        score_matrix = self.__dist_inverse(score_matrix)
        # print("----", score_matrix)
        # construct graph for matching
        G = nx.Graph()
        edges_list = []
        # print(self.objects, new_positions)
        keys_matchs = {}
        for i, a in enumerate(self.objects.keys()):
            for j, b in enumerate(new_positions):
                keys_matchs[tuple([a]+b)] = score_matrix[i][j]
                edges_list.append((a, tuple(b), {"weight": score_matrix[i][j]}))
        G.add_edges_from(edges_list)
        matched = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False, weight="weight")
        G.clear()
        # print("----matched:", matched, type(matched))
        paired = {tuple(w): False for w in new_positions}
        for match in matched:
            # print(type(match[0]), type(match[1]))
            if isinstance(match[0], tuple):
                self.objects[match[1]].update(self.__centroid(match[0]))
                paired[tuple(match[0])] = True
            else:
                self.objects[match[0]].update(self.__centroid(match[1]))
                paired[tuple(match[1])] = True

        for rect, yes in paired.items():
            if not yes:
                self.objects[self.ids] = Object(self.__centroid(rect))
                self.ids += 1
        del_these = []
        for i, t in self.objects.items():
            if now - t.last_seen > self.drop_unseen_thresh:
                del_these.append(i)
        for i in del_these:
            # print("------------- dropped: ", i)
            del self.objects[i]
        # print(self.objects)
        return self.objects, cent_rect
