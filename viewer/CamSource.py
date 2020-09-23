import os
import time
import numpy as np
import cv2
from sklearn.cluster import AgglomerativeClustering
from .pyimagesearch_centroidtracker import CentroidTracker
from .TrackingAlgo import Tracker
from .CentroidTrackerInOut import CentroidTracker as CentroidTrackerInOut
from collections import OrderedDict
import math
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from itertools import combinations


class CamSource:
    def __init__(self, cam_id, cam_name, cam_address, cam_type, cam_action, group=None, cam_area="",
                 transformationMatrix=None, dist_thresh=None, x_thresh=None, y_thresh=None):
        self.cam_id = cam_id
        if "entry_exit" in cam_action:
            self.inside = int(cam_name.split("-")[-1])
            cam_name = "-".join(cam_name.split("-")[:-1])
        else:
            self.inside = 0
        self.cam_area = cam_area
        self.cam_name = cam_name
        self.cam_address = cam_address
        self.cam_type = cam_type
        self.cam_action = cam_action
        self.people = []
        self.agg_cluster = AgglomerativeClustering()
        self.crowd = None
        self.crowd_time = None
        self.ct = CentroidTracker()
        self.ct_sitting = CentroidTracker()
        self.tracking_keys = OrderedDict()
        self.crowd_tracking_keys = OrderedDict()
        if x_thresh is not None and y_thresh is not None:
            self.ct_in = CentroidTrackerInOut(x_th=x_thresh, y_th=y_thresh)
        else:
            self.ct_in = CentroidTrackerInOut(x_th=0, y_th=0)
        self.in_tracking_keys = OrderedDict()
        self.too_many_count = 0
        self.too_few_count = 0
        self.stored_number = False
        self.group = group
        self.in_door = []
        self.out_door = []
        self.total_in_room = 0
        self.pts4 = []
        self.door = None
        self.transformationMatrix = transformationMatrix
        self.dist_thresh = dist_thresh

    @staticmethod
    def avg_distance(p1, p2):
        w1 = (p1[2] - p1[0])
        w2 = (p2[2] - p2[0])
        return abs(w2 + w1) / 2

    @staticmethod
    def centroid(p):
        x1, y1, x2, y2 = p
        return x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2

    @staticmethod
    def distance(c1, c2):
        x1, y1 = c1
        x2, y2 = c2
        return abs(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    def set_polygon(self, x):
        self.door = Polygon(x)

    def is_inside(self, pt):
        if self.door is not None:
            return self.door.contains(Point([pt[0], pt[1]]))
        else:
            return True

    def update_in_out(self, people):
        people = people["person"]
        y = self.ct_in.update(people)
        change = 0
        flag = False
        for key, val in y.items():
            if not self.is_inside(val[0]):
                continue
            flag = True
            if str(val[1]) != "0":
                if str(val[1]) == "in":
                    if key not in self.in_door:
                        self.in_door.append(key)
                        change += 1
                elif str(val[1]) == "out":
                    if key not in self.out_door:
                        self.out_door.append(key)
                        change -= 1
        if not flag:
            self.in_door = []
            self.out_door = []
        return y, change

    def update_people(self, people, max_time, max_number):
        yolo_out = {}
        now = time.time()
        safe_dist_violators = {tuple(k): False for k in people}
        count_violation, time_violation = False, False
        n_time, n_people, n_safe = 0, 0, 0
        y = self.mt.update(people)
        # print(y)
        x, cent_rect = y
        n_people = len(people)
        if n_people > max_number:
            count_violation = True
        for person, val in safe_dist_violators.items():
            if val:
                n_safe += 1
        keys = [w for w in cent_rect.keys()]
        for k, v in x.items():
            # if now - v[1] > max_time:
            if now - v.first_seen > max_time:
                time_violation = True
                if tuple(v.pos) in keys:
                    yolo_out[cent_rect[tuple(v.pos)]] = True, k
                    n_time += 1
            else:
                if tuple(v.pos) in keys:
                    yolo_out[cent_rect[tuple(v.pos)]] = False, k
        # print(n_safe)
        return count_violation, n_people, time_violation, n_time, n_safe, yolo_out

    @staticmethod
    def get_combinations(arr):
        return combinations(arr, 2)

    @staticmethod
    def compute_point_perspective_transformation(matrix, list_frame_points):
        list_frame_points = [((x1+x2)/2, y2) for (x1,y1,x2,y2) in list_frame_points]
        # Compute the new coordinates of our points
        list_points_to_detect = np.float32(list_frame_points).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
        # Loop over the points and add them to the list that will be returned
        transformed_points_list = list()
        for i in range(0, transformed_points.shape[0]):
            transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
        return transformed_points_list

    @staticmethod
    def euclidean_dist(a, b):
        return abs(math.sqrt((a[0] - b[0]) ** 2 + (b[1] - a[1]) ** 2))

    def update_crowds(self, meh, frame, people_number=3, max_time=10):
        # print("here")
        people = OrderedDict()
        cent_co = OrderedDict()
        flag = False
        result = {}
        end_violation = []
        present_new_violate = []
        now = time.time()
        for x1, y1, x2, y2 in meh:
            people[self.centroid([x1, y1, x2, y2])] = (x2 - x1, y2 - y1)
            cent_co[self.centroid([x1, y1, x2, y2])] = [x1, y1, x2, y2]
        safe_distance = {}
        social_distancing_out = {}
        # print(people.items())
        combos = self.get_combinations(range(len(people.keys())))
        temp_people = OrderedDict()
        map_transformed = {}
        if self.transformationMatrix is not None and len(people.keys()) > 0:
            # print(self.transformationMatrix, type(self.transformationMatrix))
            transformed_pts = self.compute_point_perspective_transformation(self.transformationMatrix, meh)
            transformed_pts = np.array(transformed_pts)
            transformed_pts /= 2
            for i, k in enumerate(people.keys()):
                temp_people[k] = transformed_pts[i]
                map_transformed[tuple(transformed_pts[i])] = meh[i]
        peop_dict = list(people.items())
        for pair in combos:
            p1 = peop_dict[pair[0]]
            p2 = peop_dict[pair[1]]
            p1k = p1[0]
            p1v = p1[1]
            p2k = p2[0]
            p2v = p2[1]
            if self.transformationMatrix is None:
                if self.distance(p1k, p2k) < 2 * abs((p2v[0] + p1v[0]) / 2):
                    safe_distance[p1k] = False
                    safe_distance[p2k] = False
            else:
                p1v = temp_people[p1k]
                p2v = temp_people[p2k]
                d = self.euclidean_dist(p1v, p2v)
                # print(pair, "d: ", d)
                frame_width = frame.shape[1]
                if d < self.dist_thresh*frame_width:
                    safe_distance[p1k] = False
                    safe_distance[p2k] = False
                    print("violated")
                    social_distancing_out[tuple(map_transformed[tuple(p1v)])] = False
                    social_distancing_out[tuple(map_transformed[tuple(p2v)])] = False
        cluster = {}
        cluster_in = []
        for person in safe_distance.keys():
            cluster_in.append(person)
        if len(cluster_in) > 0:
            maps = self.agg_cluster.fit(cluster_in).labels_
            for i, person in enumerate(safe_distance.keys()):
                if maps[i] not in cluster.keys():
                    cluster[maps[i]] = []
                cluster[maps[i]].append(cent_co[person])
            if len(cluster) == 0:
                return result, flag, end_violation, str(now), present_new_violate, social_distancing_out
            crowds = []
            for cluster_n, persons in cluster.items():
                if len(persons) >= people_number:
                    crowds.append(cluster[cluster_n])
            crowds = [[min([w[0] for w in crowd]), min([w[1] for w in crowd]),
                       max([w[2] for w in crowd]), max([w[3] for w in crowd])] for crowd in crowds]
            outs_track = self.ct.update(crowds)
            x, cent_rect = outs_track
            now = time.time()
            result = {}
            for k, v in x.items():
                # print(cent_rect, v, k)
                if now - v[1] > max_time:
                    result[cent_rect[tuple(v[0])]] = True
                    if k not in self.crowd_tracking_keys.keys():
                        print("yep ", k)
                        self.crowd_tracking_keys[k] = [now, now]
                        flag = True
                        present_new_violate.append(now)
                    else:
                        self.crowd_tracking_keys[k][0] = now
                else:
                    result[cent_rect[tuple(v[0])]] = False
            del_ = []
            for k, v in self.crowd_tracking_keys.items():
                if now - v[0] > 20:
                    print("this crowd dispersed")
                    end_violation.append(self.crowd_tracking_keys[k])
                    del_.append(k)
            for i in del_:
                del self.crowd_tracking_keys[i]
            return result, flag, end_violation, str(now), present_new_violate, social_distancing_out
        else:
            del_ = []
            for k, v in self.crowd_tracking_keys.items():
                if now - v[0] > 20:
                    print("this crowd dispersed")
                    end_violation.append(self.crowd_tracking_keys[k])
                    del_.append(k)
            for i in del_:
                del self.crowd_tracking_keys[i]
            return result, flag, end_violation, str(now), present_new_violate, social_distancing_out

    def update_sitting(self, meh, max_time=10):
        people = OrderedDict()
        cent_co = OrderedDict()
        present_new_violate = []
        end_violation = []
        for x1, y1, x2, y2 in meh:
            cent_co[self.centroid([x1, y1, x2, y2])] = [x1, y1, x2, y2]
        # print(people.items())
        flag = False
        outs_track = self.ct_sitting.update(meh)
        x, cent_rect = outs_track
        now = time.time()
        result = {}
        for k, v in x.items():
            # print(cent_rect, v, k)
            if now - v[1] > max_time:
                result[cent_rect[tuple(v[0])]] = True
                if k not in self.tracking_keys:
                    self.tracking_keys[k] = [now, now]
                    flag = True
                    present_new_violate.append(now)
                else:
                    self.tracking_keys[k][0] = now
            else:
                result[cent_rect[tuple(v[0])]] = False
        del_ = []
        for k, v in self.tracking_keys.items():
            if now - v[1] > 60:
                end_violation.append(self.tracking_keys[k])
                del_.append(k)
        for i in del_:
            del self.tracking_keys[i]
        return result, flag, end_violation, str(now), present_new_violate
