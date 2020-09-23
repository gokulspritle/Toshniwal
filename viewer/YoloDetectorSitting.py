import cv2
import numpy as np
import networkx as nx
import math


class YoloDetectorSitting:
    def __init__(self, model, classes):
        self.net = cv2.dnn.readNetFromDarknet(model + ".cfg", model + ".weights")
        self.classes = classes
        self.layer_names = self.net.getLayerNames()
        self.outputlayers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # --static method that returns the distance between two rectangles (l,t,b,r) with more weighing to y distance
    # -- => if y |^ then dist |^ then score |> and if score |> then weight becomes that one
    @staticmethod
    def inside(per, minx, maxx, miny, maxy):
        c = [(per[0] + per[2]) / 2, (per[1] + per[3]) / 2]
        if minx < c[1] < maxx and miny < c[0] < maxy:
            return True
        return False

    def detect(self, img, restricted_area_start, restricted_area_end, restricted_area_start_v,
               restricted_area_end_v, do_restricted_area=False, conf=0.2, nms_thresh=0.3, non_max_suppression=False,
               class_conf=[], combined_nms=False):
        if len(class_conf) < len(self.classes):
            conf = [conf] * len(self.classes)
        else:
            conf = class_conf
        final_result = {"sitting": [],
                        "person": []}
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.outputlayers)
        class_ids = []
        confidences = {}
        boxes = {}
        Height, Width, _ = img.shape
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf[int(class_id)]:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - (w / 2)
                    y = center_y - (h / 2)
                    class_ids.append(class_id)
                    if self.classes[class_id] not in boxes.keys():
                        boxes[self.classes[class_id]] = []
                        confidences[self.classes[class_id]] = []
                    confidences[self.classes[class_id]].append(float(confidence))
                    boxes[self.classes[class_id]].append([int(i) for i in [x, y, x + w, y + h]])
                    # print(self.classes[class_id])
        indices = {}
        # print(boxes)
        temp_box = []
        temp_conf = []
        temp_classes = []
        if combined_nms:
            # print(confidences, boxes)
            for class_name, box in boxes.items():
                temp_box += box
                temp_classes += [class_name] * len(box)
                temp_conf += confidences[class_name]
            indices = cv2.dnn.NMSBoxes(temp_box, temp_conf, min(conf), nms_thresh)
            for i in indices:
                select = i[0]
                final_result[temp_classes[select]].append(temp_box[select])
            return final_result
        if non_max_suppression:
            for class_name, box in boxes.items():
                indices[class_name] = cv2.dnn.NMSBoxes(box, confidences[class_name], min(conf), nms_thresh)
        else:
            for class_name, box in boxes.items():
                indices[class_name] = [[w] for w in range(len(box))]
        for key, index in indices.items():
            if not (key == "person" or key == "sitting"):
                print("error")
                continue
            if key == "sitting":
                # print(index)
                pass
            # print("key: ", key, " indices: ", index)
            for i in index:
                if do_restricted_area:
                    if self.inside(boxes['sitting'][i[0]],
                                   restricted_area_start, restricted_area_end,
                                   restricted_area_start_v, restricted_area_end_v):
                        select = i[0]
                        final_result["person"].append(boxes['person'][select])
                else:
                    select = i[0]
                    # print(self.classes[class_ids[select]], self.classes[class_ids[select]])
                    final_result[key].append(boxes[key][select])
        return final_result
