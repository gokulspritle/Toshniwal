from django.shortcuts import render
from django.shortcuts import redirect
from django.http import StreamingHttpResponse
from django.http import HttpResponse, HttpResponseRedirect
import time
import numpy as np
import os
import cv2
from imutils.video import FileVideoStream, VideoStream, WebcamVideoStream
from vidgear.gears import CamGear
import threading
import plotly.graph_objects as go
import pyttsx3
import datetime
from django.core import serializers
from dateutil.parser import parse
import base64
from django.db.models import Q
import pytz
from queue import Queue
import csv
import json
from reportlab.pdfgen import canvas
from reportlab.platypus import Table
from reportlab.platypus import SimpleDocTemplate
from .CamSource import CamSource
from .YoloDetector import YoloDetector
from .YoloDetectorSitting import YoloDetectorSitting
from .models import SnapsDB, AlertsDB, CamDB, CamTransformationMatrix, IncidentReportDB
from .analytics import number_of_occurences_today, top_priority_incidents, safety_stats, productivity_stats, \
    incident_db_list, get_alerts_by_feature, get_alerts_by_area,get_alerts_by_time

# login imports
from django.contrib import messages
from django.contrib.auth.models import User, auth
from django.contrib.auth.decorators import login_required

# "C:\Program Files (x86)\IntelSWTools\openvino_2020.1.033\bin\setupvars.bat"

# ---------------- initializing global variables ------------
lock = threading.Lock()
OutFrame = None
caps = []
cams = []
class_confidences = []
base = "/".join(os.getcwd().split('\\'))
paths = {
    "cam_instructions": base + "/cam_instructions.txt",
    "operational_params": base + "/operational_parameters.txt",
    "yolo_people_path": base + "/viewer/models/yolov3-tiny-obj",
    "yolo_standing_path": base + "/viewer/models/yolov3-tiny-sitting",
    "coco_names": base + "/viewer/models/coco.names.txt",
    "map_pts_model": base + "/viewer/models/map_points.keras",
    "yolo_cabin": base + "/viewer/models/yolov3-tiny-cabin",
    "map_pts_model_wt": base + "/viewer/models/model.pb",
    "map_pts_model_cfg": base + "/viewer/models/model.pbtxt",
    "yolo_crowd_cabin": base + "/viewer/models/yolov3-tiny-admin",
}
temps = []
swap_temps = []
tz = pytz.timezone('Asia/Kolkata')
use_own_detector = True
operational_params = {}
person_detector = YoloDetector(paths["yolo_people_path"], ["person"])
sitting_detector = YoloDetectorSitting(paths["yolo_standing_path"], ["person", "sitting"])
cabin_detector = YoloDetector(paths["yolo_cabin"], ["person"])
admin_detector = YoloDetector(paths["yolo_crowd_cabin"], ["person"])
# map_pts = load_model(paths["map_pts_model"])
map_pts = cv2.dnn.readNetFromTensorflow(paths["map_pts_model_wt"])
use_opencv = True
use_vidgear = False
def_max_people = 10
def_max_time = 5
crowd_people = 3
crowd_time = def_max_time
sitting_time = def_max_time
max_people = def_max_people
max_time = def_max_time
count_violation = False
n_people = 0
n_people_total = 0
time_violation = False
n_time = 0
n_safe = 0
last_spoken_at = time.time()
last_spoken_at_crowd = time.time()
in_between_alerts_thresh = 10
sound_alerts = True
restricted_area_start = 0.3
restricted_area_end = 1.0
restricted_area_start_v = 0.0
restricted_area_end_v = 1.0
restricted_time_start = "15:00"
restricted_time_end = "17:00"
restricted_area_module = True
restricted_alert = False
send_alert = [False, False]
server_message = ""
server_message_crowd = ""
server_message_sitting = ""
pause = False
too_many_count = 0
too_few_count = 0
this_iter = False
n_group = {}
n_group_max = {}
door_x_start = 0.73
door_y_start = 0.04
door_x_end = 0.98
door_y_end = 0.7
alerts_filter = ""
x_thresh = 0.0
y_thresh = 0.0
no_view = False
refresh_show = 20
last_played = time.time()
server_message_group = ""
voice_alerts_queue = Queue(maxsize=4)


def initialize_variables():
    global operational_params, use_opencv, in_between_alerts_thresh, sound_alerts, use_vidgear, x_thresh, y_thresh
    global no_view, refresh_show
    with open(paths["operational_params"], 'r') as f:
        lines = f.readlines()
    lines = [w.strip() for w in lines]
    left = [w.split('=')[0].strip() for w in lines]
    right = [w.split('=')[1].strip() for w in lines]
    operational_params = {k: v for k, v in zip(left, right)}
    print(operational_params)
    keys_ops = [w for w in operational_params.keys()]
    use_opencv = eval(operational_params["use_opencv"])
    use_vidgear = eval(operational_params["use_vidgear"])
    x_thresh = eval(operational_params["x_thresh"])
    y_thresh = eval(operational_params["y_thresh"])
    no_view = eval(operational_params["no_view"])
    refresh_show = eval(operational_params["refresh_show"])
    if "in_between_alerts_thresh" in keys_ops:
        in_between_alerts_thresh = int(operational_params["in_between_alerts_thresh"])
    if "sound_alerts" in keys_ops:
        sound_alerts = eval(operational_params["sound_alerts"])
    with open(paths["cam_instructions"], 'r') as f:
        lines = f.readlines()
    print(lines)
    lines = [[w.strip() for w in l.split("|")] for l in lines if not l[0] == "#"]
    print("----------lines", lines)
    for line in lines:
        if line[3] == "file":
            if use_opencv:
                caps.append(cv2.VideoCapture("./viewer/videos/" + line[2]))
            elif use_vidgear:
                caps.append(CamGear("./viewer/videos/" + line[2]).start())
            else:
                caps.append(FileVideoStream("./viewer/videos/" + line[2]).start())
        elif line[3] == "webcam":
            if use_opencv:
                caps.append(cv2.VideoCapture(int(line[2])))
            elif use_vidgear:
                caps.append(CamGear("./viewer/videos/" + line[2]).start())
            else:
                caps.append(WebcamVideoStream(int(line[2])).start())
        elif line[3] == "ip":
            if use_opencv:
                caps.append(cv2.VideoCapture(line[2]))
            elif use_vidgear:
                caps.append(CamGear("./viewer/videos/" + line[2]).start())
            else:
                caps.append(VideoStream(line[2]).start())
        else:
            for cap in caps:
                print("releasing: ", cap)
                if use_opencv:
                    cap.release()
                else:
                    cap.stop()
            raise AssertionError("unknown type of stream, please check cam_instructions.txt")
        if True:
            print(line)
            if line[3] not in ["webcam", "file", "ip"]:
                raise AssertionError("unknown type of stream, please check cam_instructions.txt")
            if line[4] not in ["count", "crowd", "sitting"]:
                raise AssertionError("unknown type of camera action selected, please check cam_instructions.txt")
            cams.append(CamSource(line[0], line[1], line[2], line[3], [line[4]]))


# -------------- drawing values on images -------------------
def draw_on(frame, results):
    people = results["person"]
    for x, y, w, h in people:
        cv2.rectangle(frame, (x, y), (w, h), (32, 199, 232), thickness=2)
        cv2.putText(frame, "person", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), thickness=2)
    return frame


def draw_restricted(frame):
    global restricted_area_start, restricted_area_end, restricted_area_start_v, restricted_area_end_v
    H, W, _ = frame.shape
    start = int(restricted_area_start * H)
    end = int(restricted_area_end * H)
    # frame = cv2.line(frame, (0, start), (W, start), (255, 255, 255))
    # frame = cv2.line(frame, (0, end), (W, end), (255, 255, 255))
    start_v = int(restricted_area_start_v * W)
    end_v = int(restricted_area_end_v * W)
    # frame = cv2.line(frame, (start_v, 0), (start_v, H), (255, 255, 255))
    # frame = cv2.line(frame, (end_v, 0), (end_v, H), (255, 255, 255))
    frame = cv2.rectangle(frame, (start_v, start),
                          (end_v, end), (0, 0, 255))
    return frame


# ---------------- Analysis algorithms ----------------------

def voice_alert(say_this):
    print("hello", say_this)
    engine = pyttsx3.init()
    engine.setProperty("rate", 140)
    voices = engine.getProperty('voices')
    ids = []
    for i in voices:
        # print(i.name)
        # print(i.id)
        ids.append(i.id)
    if "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0" in ids:
        engine.setProperty("voice", "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0")
    # print("-------------- to say this:", say_this)
    engine.say(say_this)
    try:
        engine.runAndWait()
    except Exception as e:
        print(e)


def alerts_threaded():
    global voice_alerts_queue
    while True:
        if not voice_alerts_queue.empty():
            t_voice = threading.Thread(target=voice_alert, args=(voice_alerts_queue.get(),))
            t_voice.start()
            while t_voice.is_alive():
                continue


if sound_alerts:
    alert_thread = threading.Thread(target=alerts_threaded)
    alert_thread.daemon = True
    alert_thread.start()


def analyse(yolo_out, cam):
    global max_people, max_time, count_violation, n_people, time_violation, n_time, voice_alerts_queue
    global n_safe, last_spoken_at, in_between_alerts_thresh, sound_alerts
    count_violation, n_people, time_violation, n_time, n_safe, yolo_out = cam.update_people(yolo_out["person"],
                                                                                            max_time, max_people)
    now = time.time()
    if now - last_spoken_at > in_between_alerts_thresh:
        if n_time > 0 or n_people > max_people:
            if sound_alerts:
                voice_alerts_queue.put("please return to your work stations")
        last_spoken_at = now
    return yolo_out


def save_this_image_to_snaps_db(img, time_ref):
    here_img = cv2.resize(img.copy(), None, fx=0.5, fy=0.5)  # 300, 300 is the image shape in pixels
    # just to make sure that other processes don't change here_img while converting
    with lock:
        flag, encoded_image = cv2.imencode(".jpg", here_img)
        strImg = base64.b64encode(encoded_image)
    SnapsDB.objects.create(snap=strImg.decode(),
                           ref_seconds=str(time_ref))
    print("length of snap saved: ", len(strImg))


def analyse_crowds(yolo_out, cam, frame):
    global server_message_crowd, crowd_people, crowd_time, n_people_total, last_spoken_at_crowd, voice_alerts_queue
    n_people_total += len(yolo_out["person"])
    heh = cam.update_crowds(yolo_out["person"], frame, crowd_people, crowd_time)
    # print("heh", heh)
    try:
        crowds, crowd_alert, end_violation, now, prsnt_nw_vi, safe_distance = heh
    except ValueError:
        print("error")
        crowds, crowd_alert, end_violation, now, prsnt_nw_vi = heh
        safe_distance = {}
    # print(crowds)
    assigned = False
    now = time.time()
    for crowd, status in crowds.items():
        if status:
            cv2.rectangle(frame, (crowd[0], crowd[1]), (crowd[2], crowd[3]), (0, 0, 255), thickness=2)
            cv2.putText(frame, "crowd violation", (crowd[0], crowd[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),
                        thickness=2)
            assigned = True
            server_message_crowd = "Crowd Violation Detected"
            if now - last_spoken_at_crowd > in_between_alerts_thresh:
                if sound_alerts:
                    voice_alerts_queue.put("Crowd violation detected in area " + cam.cam_name)
                last_spoken_at_crowd = now
        else:
            cv2.rectangle(frame, (crowd[0], crowd[1]), (crowd[2], crowd[3]), (0, 69, 255), thickness=2)
            cv2.putText(frame, "crowd detected", (crowd[0], crowd[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 69, 255),
                        thickness=2)
            if not assigned and not server_message_crowd == "crowd violation detected":
                server_message_crowd = "crowd detected"
    if len(crowds) == 0 and server_message_crowd == "":
        server_message_crowd = "no crowds detected"
    elif len(crowds) > 0:
        if now - last_spoken_at_crowd > in_between_alerts_thresh:
            if sound_alerts:
                voice_alerts_queue.put("Crowd detected in area " + cam.cam_name)
            last_spoken_at_crowd = now
    if crowd_alert:
        print("create crowd alert")
        for ref in prsnt_nw_vi:
            AlertsDB.objects.create(ref_seconds=ref,
                                    alert_type="Crowd Gathering",
                                    alert_start_time=datetime.datetime.fromtimestamp(float(ref)),
                                    alert_end_time="processing",
                                    cam_id=cam.cam_id,
                                    cam_name=cam.cam_name,
                                    cam_type=cam.cam_type,
                                    cam_area=cam.cam_area)
            save_this_image_to_snaps_db(frame, ref)
            break
    for meh in end_violation:
        print(meh)
        AlertsDB.objects.filter(Q(ref_seconds=str(meh[1])), Q(alert_type="Crowd Gathering")).update(
            alert_end_time=datetime.datetime.fromtimestamp(float(now)))
    if len(safe_distance.keys()) > 0:
        print(safe_distance)
        for k, v in safe_distance.items():
            x, y, w, h = k
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), thickness=2)
        save_this_image_to_snaps_db(frame, now)
        AlertsDB.objects.create(ref_seconds=now,
                                alert_type="Crowd Gathering",
                                alert_start_time=datetime.datetime.fromtimestamp(float(now)),
                                alert_end_time="-",
                                cam_id=cam.cam_id,
                                cam_name=cam.cam_name,
                                cam_type=cam.cam_type,
                                cam_area=cam.cam_area)
        if sound_alerts:
            voice_alerts_queue.put("please maintain social distance in " + cam.cam_area + " for your safety")
    return frame


def analyse_sitting(yolo_out, cam, frame):
    global server_message_sitting, sitting_time, sitting_time, n_people_total
    n_people_total += len(yolo_out["sitting"])
    n_people_total += len(yolo_out["person"])
    crowds, sitting_alert, end_violation, now, prsnt_nw_vi = cam.update_sitting(yolo_out["sitting"], sitting_time)
    assigned = False
    for crowd, status in crowds.items():
        if status:
            cv2.rectangle(frame, (crowd[0], crowd[1]), (crowd[2], crowd[3]), (0, 0, 255), thickness=2)
            cv2.putText(frame, "sitting violation", (crowd[0], crowd[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),
                        thickness=2)
            assigned = True
            server_message_sitting = "sitting violation detected"
        else:
            cv2.rectangle(frame, (crowd[0], crowd[1]), (crowd[2], crowd[3]), (0, 69, 255), thickness=2)
            cv2.putText(frame, "sitting posture detected", (crowd[0], crowd[1]), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 69, 255),
                        thickness=2)
            if not assigned and not server_message_sitting == "sitting violation detected":
                server_message_sitting = "sitting posture detected"
    if len(crowds) == 0 and server_message_sitting == "":
        server_message_sitting = "no sitting violations detected"
    if sitting_alert:
        save_this_image_to_snaps_db(frame, now)
        for _ in prsnt_nw_vi:
            AlertsDB.objects.create(ref_seconds=now,
                                    alert_type="Sitting Posture",
                                    alert_start_time=datetime.datetime.fromtimestamp(float(now)),
                                    alert_end_time="processing",
                                    cam_id=cam.cam_id,
                                    cam_name=cam.cam_name,
                                    cam_type=cam.cam_type,
                                    cam_area=cam.cam_area)
    for meh in end_violation:
        AlertsDB.objects.filter(Q(ref_seconds=str(meh[1]))).update(
            alert_end_time=datetime.datetime.fromtimestamp(float(now)))

    return frame


def analyse_number(yolo_out, frame, cam):
    global max_people, count_violation, server_message, n_people, n_people_total, too_few_count, too_many_count, \
        this_iter, voice_alerts_queue, last_spoken_at
    now = str(time.time())
    people = yolo_out["person"]
    n_people = len(people)
    n_people_total += n_people
    if len(people) > max_people:
        count_violation = True
        if cam.too_many_count < 5:
            cam.too_many_count += 1
        server_message = "storeroom has exceeded size limit"
    elif not server_message == "storeroom has exceeded size limit":
        server_message = "storeroom is within limit"
        count_violation = False
    if len(people) < max_people and cam.too_many_count > 0:
        cam.too_many_count -= 1
    if cam.too_many_count >= 5 and not cam.stored_number:
        this_iter = True
        cam.stored_number = True
        save_this_image_to_snaps_db(frame, now)
        AlertsDB.objects.create(ref_seconds=now,
                                alert_type="People Count",
                                alert_start_time=datetime.datetime.fromtimestamp(float(now)),
                                alert_end_time="processing",
                                cam_id=cam.cam_id,
                                cam_name=cam.cam_name,
                                cam_type=cam.cam_type,
                                cam_area=cam.cam_area)
        if float(now) - last_spoken_at > in_between_alerts_thresh:
            if sound_alerts:
                voice_alerts_queue.put("Too many people in area " + cam.cam_name)
            last_spoken_at = float(now)
    if cam.too_many_count == 0 and cam.stored_number and this_iter:
        cam.stored_number = False
        x = AlertsDB.objects.order_by("-pk")[0]
        x.alert_end_time = datetime.datetime.fromtimestamp(float(now))
        x.save()
        this_iter = False


def get_centroids(yolo_out):
    meh = []
    for x1, y1, x2, y2 in yolo_out:
        meh.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
    return meh


def analyse_groups(g, H, W):
    global map_pts
    result = {}
    special = {}
    for group_name, items in g.items():
        if len(group_name.split('-')) > 1:
            special[group_name] = get_centroids(items)
        else:
            result[group_name] = len(items)
    # print("ok special: ", special)
    if len(special) == 2:
        first = []
        second = []
        for k, v in special.items():
            if k[-1] == 'a':
                first = v
            elif k[-1] == 'b':
                second = v
        # print("first, second: ", first, second)
        if len(first) == 0 or len(second) == 0:
            for k, v in special.items():
                result[k] = max([len(first), len(second)])
        else:
            mapped_pts = []
            for pt in first:
                map_pts.setInput(np.array([[pt[0] / W, pt[1] / H]]))
                pred = map_pts.forward()
                pred = pred[0]
                pred = [pred[0] * W, pred[1] * H]
                # print("pred: ", pred)
                mapped_pts.append(pred)
            # print("mapped pts: ", mapped_pts)
            mapped_pts = [np.array(w) for w in mapped_pts]
            second = [np.array(w) for w in second]
            for pt in mapped_pts:
                add = True
                for xy in second:
                    dist = np.linalg.norm(pt - xy)
                    # print("dist:", dist)
                    if dist < 200:
                        add = False
                        break
                if add:
                    second.append(pt)
            for k, v in special.items():
                result[k] = max([len(first), len(second)])
    return result


def analyse_entrance(yolo_out, cam, frame):
    H, W, _ = frame.shape
    global door_x_start, door_x_end, door_y_end, door_y_start
    y, change = cam.update_in_out(yolo_out)
    xs, xe, ys, ye = [int(w) for w in [door_x_start * W, door_x_end * W, door_y_start * H, door_y_end * H]]
    four_points = [(0.7239583333333334 * W, 0.3814814814814815 * H),
                   (0.7677083333333333 * W, 0.02962962962962963 * H),
                   (0.9645833333333333 * W, 0.1537037037037037 * H),
                   (0.9 * W, 0.5314814814814814 * H), ]
    if cam.door is None:
        print("set door points: ", four_points)
        cam.set_polygon(four_points)
    # print(xs, ys, xe, ye)
    for key, val in y.items():
        # print(key, val)
        color = np.random.rand(3, )
        cv2.putText(frame, str(key) + " angle: " + str(val[1])[:4], (val[0][0], val[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)
        cv2.circle(frame, (val[0][0], val[0][1]), 10, (0, 255, 0), thickness=-1)
        cv2.line(frame, (val[0][0], val[0][1]), (val[2][0], val[2][1]), color, thickness=2)
        # cv2.rectangle(frame, (xs, ys), (xe, ye), (0, 255, 0))
        for inds in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            cv2.line(frame, tuple([int(w) for w in four_points[inds[0]]]),
                     tuple([int(w) for w in four_points[inds[1]]]),
                     (0, 0, 255))
    if not change == 0:
        print("change: ", change)
    return frame, change


# -------------- computer vision algorithms -----------------

# # ----- create images in n X rows matrix ------------------

def stitch_frames(out_frames):
    global cams
    x = len(out_frames)
    if x == 1 and len(cams) == 1:
        return out_frames[0]
    if not x % 2 == 0:
        x += 1
    rows = []
    res = []
    for i in range(0, x, 2):
        if i + 1 <= len(out_frames) - 1:
            out_frames[i + 0] = cv2.resize(out_frames[i + 0].copy(), (640, 480))
            out_frames[i + 1] = cv2.resize(out_frames[i + 1].copy(), (640, 480))
            rows.append(np.concatenate(
                (out_frames[i + 0], out_frames[i + 1]), axis=1))
        else:
            out_frames[i + 0] = cv2.resize(out_frames[i + 0].copy(), (640, 480))
            rows.append(np.concatenate((out_frames[i + 0], np.zeros(
                (out_frames[i + 0].shape[0], out_frames[i + 0].shape[1], 3), np.uint8)), axis=1))
    for row in rows:
        if not len(res) == 0:
            res = np.concatenate((res, row), axis=0)
        else:
            res = row
    return res


def skip_n(n, cap):
    global use_opencv
    if use_opencv:
        for i in range(n):
            ret, frame = cap.read()
        return ret, frame
    else:
        for i in range(n):
            frame = cap.read()
        return frame is not None, frame


def re_init_and_skip_n(n, cap, index_of_cam):
    global cams, caps, use_opencv, use_vidgear
    ret, frame = skip_n(n, cap)
    if not ret:
        if use_opencv:
            caps[index_of_cam].release()
        else:
            caps[index_of_cam].stop()
        print("no frames error!\ntrying to reinitialize cap... ", "\npresent_caps: ", caps)
        c = cams[index_of_cam]
        if use_opencv:
            if c.cam_type == "file":
                caps[index_of_cam] = cv2.VideoCapture("./viewer/videos/" + c.cam_address)
            else:
                caps[index_of_cam] = cv2.VideoCapture(c.cam_address)
        elif use_vidgear:
            if c.cam_type == "file":
                caps[index_of_cam] = CamGear("./viewer/videos/" + c.cam_address).start()
            else:
                caps[index_of_cam] = CamGear(c.cam_address).start()
        else:
            if c.cam_type == "file":
                caps[index_of_cam] = FileVideoStream("./viewer/videos/" + c.cam_address).start()
            else:
                caps[index_of_cam] = VideoStream(c.cam_address).start()
        print("reinitialized captures: ", caps)
        return skip_n(n, caps[index_of_cam])
    return ret, frame


# # ------ threaded function to update OutFrame -------------
def update_frames():
    global cams, OutFrame, lock, caps, temps, fd, border_thresh, swap_temps, no_view, refresh_show
    global use_own_detector, use_opencv, pause, n_people_total, n_group, cabin_detector, last_played
    global restricted_area_start, restricted_area_end, restricted_area_start_v, restricted_area_end_v
    global server_message_group, last_spoken_at, admin_detector, voice_alerts_queue
    if True:
        while True:
            while pause:
                # print("paused")
                pass
            if len(temps) > 0:
                swap_temps = temps
            g_temp = {}
            n_temp = 0
            H, W = 0, 0
            for index_of_cam, (cap, cam) in enumerate(zip(caps, cams)):
                n_tempx = 0
                if pause:
                    print("paused")
                    continue
                if use_opencv:
                    ret, frame = re_init_and_skip_n(8, cap, index_of_cam)
                else:
                    ret, frame = re_init_and_skip_n(4, cap, index_of_cam)
                if not ret:
                    print("no frame error!")
                    continue
                frame = cv2.resize(frame.copy(), None, fx=0.8, fy=0.8)
                pass_it = frame.copy()
                H, W, _ = frame.shape
                yolo_out = {"person": []}
                if "entry_exit" in cam.cam_action:
                    yolo_out = cabin_detector.detect(pass_it, restricted_area_start * H, restricted_area_end * H,
                                                     restricted_area_start_v * W,
                                                     restricted_area_end_v * W, non_max_suppression=True,
                                                     nms_thresh=0.7,
                                                     conf=0.1, do_restricted_area=False)
                    n_tempx = yolo_out["person"]
                    frame, change = analyse_entrance(yolo_out, cam, frame)
                    if "entry_exit" not in n_group.keys():
                        n_group["entry_exit"] = cam.inside
                    n_group["entry_exit"] += change
                    if "crowd" in cam.cam_action:
                        frame = analyse_crowds(yolo_out, cam, frame)
                if "count" in cam.cam_action:
                    if cam.cam_name == "storeroom":
                        yolo_out = person_detector.detect(pass_it, restricted_area_start * H, restricted_area_end * H,
                                                          restricted_area_start_v * W,
                                                          restricted_area_end_v * W, non_max_suppression=True,
                                                          nms_thresh=0.3,
                                                          conf=0.1, do_restricted_area=True)
                        n_tempx = yolo_out["person"]
                        analyse_number(yolo_out, frame, cam)
                        frame = draw_restricted(frame)
                    else:
                        yolo_out = person_detector.detect(pass_it, restricted_area_start * H, restricted_area_end * H,
                                                          restricted_area_start_v * W,
                                                          restricted_area_end_v * W, non_max_suppression=True,
                                                          nms_thresh=0.3,
                                                          conf=0.1, do_restricted_area=False)
                        n_tempx = yolo_out["person"]
                        # analyse_number(yolo_out, frame, cam)
                    if cam.group is not None:
                        if not cam.group in g_temp.keys():
                            g_temp[cam.group] = []
                        g_temp[cam.group] = n_tempx
                frame = draw_on(frame, yolo_out)
                if "crowd" in cam.cam_action and "entry_exit" not in cam.cam_action:
                    if not cam.cam_area == "admin block":
                        yolo_out = person_detector.detect(pass_it, restricted_area_start * H, restricted_area_end * H,
                                                          restricted_area_start_v * W,
                                                          restricted_area_end_v * W, non_max_suppression=True,
                                                          nms_thresh=0.5,
                                                          conf=0.1, do_restricted_area=False)
                    else:
                        yolo_out = admin_detector.detect(pass_it, restricted_area_start * H, restricted_area_end * H,
                                                         restricted_area_start_v * W,
                                                         restricted_area_end_v * W, non_max_suppression=True,
                                                         nms_thresh=0.5,
                                                         conf=0.1, do_restricted_area=False)
                    frame = analyse_crowds(yolo_out, cam, frame)
                    n_tempx = yolo_out["person"]
                    if cam.group is not None:
                        if not cam.group in g_temp.keys():
                            g_temp[cam.group] = []
                        g_temp[cam.group] = yolo_out["person"]
                frame = draw_on(frame, yolo_out)
                if "sitting" in cam.cam_action:
                    yolo_out = sitting_detector.detect(pass_it, restricted_area_start * H, restricted_area_end * H,
                                                       restricted_area_start_v * W,
                                                       restricted_area_end_v * W, non_max_suppression=True,
                                                       nms_thresh=0.3,
                                                       conf=0.1, do_restricted_area=False, combined_nms=True)
                    # print(yolo_out)
                    analyse_sitting(yolo_out, cam, frame)
                    n_tempx = yolo_out["person"] + yolo_out["sitting"]
                    if cam.group is not None:
                        if not cam.group in g_temp.keys():
                            g_temp[cam.group] = []
                        g_temp[cam.group] = n_tempx
                n_temp += len(n_tempx)
                frame = draw_on(frame, yolo_out)
                if "count" in cam.cam_action and "storeroom" in cam.cam_name.split("-"):
                    frame = draw_restricted(frame)
                if len(temps) <= index_of_cam:
                    temps.append(frame)
                else:
                    temps[index_of_cam] = frame
            n_people_total = n_temp
            n_group.update(analyse_groups(g_temp.copy(), H, W))
            server_message_group = ""
            all_here = {}
            for k, v in n_group.items():
                if len(k.split("-")) == 1:
                    all_here[k] = v
                elif len(k.split("-")) == 2 and (k[-1] == "a"):
                    k = k.split("-")[0]
                    if k not in all_here.keys():
                        all_here[k] = 0
                    all_here[k] += max(n_group[k + "-a"], n_group[k + "-b"])
            for k, v in all_here.items():
                if k in n_group_max.keys():
                    if v < n_group_max[k]:
                        server_message_group += "\nNumber of people in Group " + str(k) + " : " + str(v)
                    else:
                        now = time.time()
                        if now - last_spoken_at > in_between_alerts_thresh:
                            if sound_alerts:
                                voice_alerts_queue.put("Too many people in area " + str(k))
                            last_spoken_at = now
                            if OutFrame is not None:
                                time_ref = time.time()
                                save_this_image_to_snaps_db(OutFrame, time_ref)
                                print("ok so this:", k, [cam.group for cam in cams],
                                      [cam.cam_id for cam in cams if cam.group == k])
                                AlertsDB.objects.create(ref_seconds=time_ref,
                                                        alert_type="People Count",
                                                        alert_start_time=datetime.datetime.fromtimestamp(
                                                            float(now)),
                                                        alert_end_time="-",
                                                        cam_id="".join(
                                                            [cam.cam_id for cam in cams if cam.group == k]),
                                                        cam_name="".join(
                                                            [cam.cam_name for cam in cams if cam.group == k]),
                                                        cam_type="ip",
                                                        cam_area="".join(
                                                            [cam.cam_area for cam in cams if cam.group == k]))
                                print("saved!")
                        server_message_group += "\nExceeded Number of people in Group " + str(k) + " : " + str(v)
                else:
                    # print(k, n_group_max)
                    server_message_group += "\nNo Max defined Number of people in Group " + str(k) + " : " + str(v)
            if no_view:
                # print("no view")
                # print(last_played, refresh_show, time.time()-last_played)
                if time.time() - last_played < refresh_show:
                    # print("updated")
                    with lock:
                        if len(temps) == 0:
                            print("no temps")
                            temps = swap_temps
                        OutFrame = stitch_frames(temps)
            else:
                with lock:
                    if len(temps) == 0:
                        print("no temps")
                        temps = swap_temps
                    OutFrame = stitch_frames(temps)
    # except Exception as e:
    #     print(e)


# --------------- code run at start -------------------------
initialize_variables()
t = threading.Thread(target=update_frames)
t.daemon = True
t.start()


# ------------ views and requests start here ----------------
def get_stream_html():
    global cams
    length = len(cams)
    print("length of cams: ", length, [i for i in range(length)])
    if length % 2 == 1:
        length += 1
    return [i for i in range(length)][:-1]


@login_required(login_url="/viewer/login")
def index(request):
    global max_people, max_time, restricted_area_start, restricted_area_end, restricted_time_start, restricted_time_end
    global sitting_time, crowd_people, crowd_time, n_group_max, last_played
    here_group_limit = ""
    last_played = time.time()
    for k, v in n_group_max.items():
        here_group_limit += str(k) + "=" + str(v) + ","
    return render(request, 'viewer/index.html', {"max_people": max_people,
                                                 "max_time": max_time,
                                                 "restricted_area_start": restricted_area_start * 100,
                                                 "restricted_area_end": restricted_area_end * 100,
                                                 "restricted_area_start_v": restricted_area_start_v * 100,
                                                 "restricted_area_end_v": restricted_area_end_v * 100,
                                                 "restricted_time_start": restricted_time_start,
                                                 "restricted_time_end": restricted_time_end,
                                                 "sitting_time": sitting_time,
                                                 "crowd_people": crowd_people,
                                                 "crowd_time": crowd_time,
                                                 "n_group_max": here_group_limit, })


@login_required(login_url="/viewer/login")
def index_new(request):
    global max_people, max_time, restricted_area_start, restricted_area_end, restricted_time_start, restricted_time_end
    global sitting_time, crowd_people, crowd_time, n_group_max, last_played
    here_group_limit = ""
    last_played = time.time()
    for k, v in n_group_max.items():
        here_group_limit += str(k) + "=" + str(v) + ","
    return render(request, 'viewer/index_new.html', {"max_people": max_people,
                                                     "max_time": max_time,
                                                     "restricted_area_start": restricted_area_start * 100,
                                                     "restricted_area_end": restricted_area_end * 100,
                                                     "restricted_area_start_v": restricted_area_start_v * 100,
                                                     "restricted_area_end_v": restricted_area_end_v * 100,
                                                     "restricted_time_start": restricted_time_start,
                                                     "restricted_time_end": restricted_time_end,
                                                     "sitting_time": sitting_time,
                                                     "crowd_people": crowd_people,
                                                     "crowd_time": crowd_time,
                                                     "n_group_max": here_group_limit,
                                                     "stream_html": get_stream_html()})


@login_required(login_url="/viewer/login")
def dashboard(request):
    return render(request, 'viewer/dashboard.html')


@login_required(login_url='/viewer/login')
def analytics(request):
    # data for 1 week
    today = datetime.datetime.today()
    EachDayAlerts = {}
    for i in range(6, -1, -1):
        curr = today - datetime.timedelta(days=i)
        curr_st = str(curr).split(" ")[0] + " 00:00:00"
        curr_end = str(curr).split(" ")[0] + " 23:59:59"
        EachDayAlerts[curr_st.split(" ")[0]] = [len(AlertsDB.objects.filter(
            Q(alert_start_time__range=[curr_st, curr_end]), alert_type="Crowd Gathering")),
            len(AlertsDB.objects.filter(
                Q(alert_start_time__range=[curr_st, curr_end]), alert_type="People Count")),
            len(AlertsDB.objects.filter(
                Q(alert_start_time__range=[curr_st, curr_end]), alert_type="Sitting Posture")), ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[w for w in EachDayAlerts.keys()], y=[v[2] for v in EachDayAlerts.values()],
                             mode='lines+markers',
                             name='Sitting Posture'))
    fig.add_trace(go.Scatter(x=[w for w in EachDayAlerts.keys()], y=[v[0] for v in EachDayAlerts.values()],
                             mode='lines+markers',
                             name='Crowd Gathering'))
    fig.add_trace(go.Scatter(x=[w for w in EachDayAlerts.keys()], y=[v[1] for v in EachDayAlerts.values()],
                             mode='lines+markers',
                             name='People Count'))
    fig.update_layout(title='Alert log',
                      xaxis_title='Day',
                      yaxis_title='Total_alerts')
    # fig.show()
    x = fig.to_html(include_plotlyjs='cdn')
    chart = x.split("<body>")[-1].split("</body>")[0]
    return render(request, 'viewer/analytics.html', {"chart": chart})


def send_frame():
    while True:
        with lock:
            global OutFrame
            if OutFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", OutFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def stream_cam(request):
    try:
        return StreamingHttpResponse(send_frame(), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(e)


def send_frame_new(stream_number):
    while True:
        with lock:
            global temps
            if len(temps) <= stream_number:
                continue
            if temps[stream_number] is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", cv2.resize(temps[stream_number], (400, 400)))
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def stream_cam_multiple(request, stream_number):
    try:
        return StreamingHttpResponse(send_frame_new(stream_number),
                                     content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(e, stream_number)


def update_data(request):
    if request.method == 'POST':
        global max_people, crowd_people, crowd_time, sitting_time, n_group_max
        dat = dict(request.POST)
        try:
            mp = int(dat["max_people"][0])
            max_people = mp
            cp = int(dat["crowd_people"][0])
            crowd_people = cp
            ct = int(dat["crowd_time"][0])
            crowd_time = ct
            st = int(dat["sitting_time"][0])
            sitting_time = st
            gm = str(dat["group_max"][0])
            gm = gm.split(',')
            n_group_max = {}
            for element in gm:
                l, r = element.split("=")
                n_group_max[l.strip()] = int(r)
            print("n_group_max: ", n_group_max)
        except Exception as e:
            print(e) 
        data = {
                "max_people": max_people,
                "crowd_people":crowd_people,
                "crowd_time": crowd_time,
                "sitting_time": sitting_time,
                "n_group_max" : n_group_max,
                }
        return HttpResponse(json.dumps(data), content_type='application/json')
        #return HttpResponseRedirect('/viewer/')


def update_restrict_data(request):
    if request.method == "POST":
        global restricted_area_start, restricted_area_end
        global restricted_area_start_v, restricted_area_end_v
        dat = dict(request.POST)
        print(dat)
        restricted_area_start = int(dat["restricted_area_start"][0]) / 100
        restricted_area_end = int(dat["restricted_area_end"][0]) / 100
        restricted_area_start_v = int(dat["restricted_area_start_v"][0]) / 100
        restricted_area_end_v = int(dat["restricted_area_end_v"][0]) / 100
        data = {
                "restricted_area_start": restricted_area_start,
                "restricted_area_end":restricted_area_end,
                "restricted_area_start_v": restricted_area_start_v,
                "sitting_time": sitting_time,
                "restricted_area_end_v" : restricted_area_end_v,
                }
        return HttpResponse(json.dumps(data), content_type='application/json')


def get_data(request):
    if request.headers.get('accept') == 'text/event-stream':
        def events():
            global server_message, n_people, n_people_total, server_message_crowd, server_message_sitting, last_spoken_at
            global n_group, n_group_max, OutFrame, server_message_group
            if True:
                if server_message == "":
                    server_message = ""
                if server_message_crowd == "":
                    server_message_crowd = ""
                data = ["'" + str(server_message.upper()) + "'",
                        "'" + str(n_people_total) + "'",
                        "'" + str(server_message_crowd.upper()) + "'",
                        "'" + str(server_message_sitting.upper()) + "'",
                        "'" + str(server_message_group.upper()) + "'"]
                server_message = ""
                server_message_crowd = ""
                yield "data: {data_out}\n\n".format(data_out=data)
            # except Exception as e:
            #     print(e)

        return HttpResponse(events(), content_type='text/event-stream')


def return_cam_list():
    objs = CamDB.objects.all()
    global cams
    ret = []
    if len(objs) > 0:
        here = objs
    else:
        here = cams.copy()
    for cam in here:
        cam_type_dict = {
            "ip": "",
            "file": "",
            "webcam": ""
        }
        meh = cam.group
        if meh is None:
            meh = "None"
        if isinstance(cam.cam_action, list):
            cxn = cam.cam_action
        else:
            cxn = cam.cam_action.split(",")
        cam_type_dict[cam.cam_type] = " selected "
        select_cam_type = ""
        for k, v in cam_type_dict.items():
            select_cam_type += '<option value="' + k.capitalize() + '"' + v + '>' + k.capitalize() + '</option>'
        ret.append([cam.cam_address, select_cam_type, cam.cam_name, cam.cam_area, cam.cam_id, ",".join(cxn), meh])
    return [[w for w in x] for x in ret]


def validate_this(dat):
    cam_urls = [w for w in dat["cam_url"]]
    cam_types = [w.lower() for w in dat["cam_type"]]
    cam_names = [w.lower() for w in dat["cam_name"]]
    cam_ids = [w.lower() for w in dat["cam_id"]]
    cam_functions = dat["cam_function"]
    for x in cam_types:
        if x not in ["ip", "webcam", "file"]:
            return False
    for x in cam_functions:
        x = [w.lower() for w in x.split(",")]
        for y in x:
            if y not in ["crowd", "count", "sitting", "entry_exit"]:
                return False
    return True


def assign_these(dat):
    global cams, caps, use_opencv, use_vidgear, x_thresh, y_thresh
    cams = []
    CamDB.objects.all().delete()
    for cap in caps:
        if use_opencv:
            print("use opencv")
            cap.release()
        elif use_vidgear:
            print("using vidgear")
            cap.stop()
        else:
            print("use imutils")
    caps = []
    cam_urls = [w for w in dat["cam_url"]]
    cam_types = [w.lower() for w in dat["cam_type"]]
    cam_names = [w.lower() for w in dat["cam_name"]]
    cam_ids = [w.lower() for w in dat["cam_id"]]
    cam_functions = [[w.lower() for w in x.split(",")] for x in dat["cam_function"]]
    cam_groups = [w for w in dat["cam_group"]]
    print("cam functions: ", cam_functions)
    cam_areas = [w for w in dat["cam_area"]]
    camTransformationMatrix = []
    camDistThresh = []
    for uri in cam_urls:
        tm = CamTransformationMatrix.objects.filter(Q(cam_address=uri)).order_by("-pk")
        if len(tm) > 0:
            tm = tm[0]
            camTransformationMatrix.append(np.array(eval(tm.transformationMatrix)))
            camDistThresh.append(float(tm.distanceThresh))
        else:
            camTransformationMatrix.append(None)
            camDistThresh.append(None)
    for a, b, c, d, e, f, g, h, i in zip(cam_urls, cam_types, cam_names, cam_ids, cam_functions,
                                         cam_groups, cam_areas, camTransformationMatrix, camDistThresh):
        if len(a) == 0:
            continue
        if (len(f) == 0) or (f.lower() == "none"):
            f = None
        print(x_thresh, y_thresh, "assign these")
        cams.append(CamSource(cam_id=d, cam_name=c, cam_address=a, cam_type=b, cam_action=e, group=f, cam_area=g,
                              transformationMatrix=h, dist_thresh=i, x_thresh=x_thresh, y_thresh=y_thresh))
        if use_opencv:
            if b == "file":
                caps.append(cv2.VideoCapture("./viewer/videos/" + a))
            else:
                caps.append(cv2.VideoCapture(a))
        elif use_vidgear:
            if b == "file":
                caps.append(CamGear("./viewer/videos/" + a).start())
            else:
                caps.append(CamGear(a).start())
        else:
            if b == "file":
                caps.append(FileVideoStream("./viewer/videos/" + a).start())
            else:
                caps.append(VideoStream(a).start())
        e = ",".join(e)
        # print("e: ", e)
        if f is None:
            f = "None"
        x = CamDB(cam_address=a, cam_type=b, cam_name=c, cam_id=d, cam_action=e, group=f, cam_area=g)
        x.save()


@login_required(login_url="/viewer/login")
def cam_settings(request):
    if request.method == "POST":
        global pause
        pause = True
        dat = dict(request.POST)
        print(dat)
        if validate_this(dat):
            assign_these(dat)
            print("assigned")
        else:
            print("not assigned")
            return HttpResponseRedirect("/viewer/CamSettings")
        pause = False
        return HttpResponseRedirect("/viewer/")
    return render(request, "viewer/cam_settings.html", {"cam_list": return_cam_list()})


@login_required(login_url="/viewer/login")
def management_dashboard(request):
    incidents = top_priority_incidents(5)
    return render(request, "viewer/management.html", {"priorities": incidents})


def report_incident(request, reference_second):
    incidents = IncidentReportDB.objects.filter(Q(ref_seconds=reference_second))
    alert_type = AlertsDB.objects.filter(Q(ref_seconds=reference_second)).order_by("-pk")[0]
    #print("incidents: ", len(incidents))
    search_value = request.GET.get("check_this", None)
    #print("search", search_value)
    if search_value is not None:
        severity = request.GET.get("severity", None)
        spotted_on = request.GET.get("spotted_on", None)
        cam_area = request.GET.get("cam_area", None)
        ref_sec = request.GET.get("ref_sec", None).split("/")[-1].split('?')[0]
        comment = request.GET.get("comment", None)
        occurrence_count = request.GET.get("occurrence_count", None)
        #print(severity, spotted_on, cam_area, ref_sec, comment, occurrence_count)
        if search_value == "register_new":
            change_permission = User.objects.get(
                id=request.session['_auth_user_id']).has_perm('viewer.change_IncidentReportDB')
            if not change_permission:
                print("the user doesn't have permission!")
            if len(incidents) == 0 and change_permission:
                x = IncidentReportDB(severity=severity, spotted_on=spotted_on, ref_seconds=ref_sec,
                                     area=cam_area, comments=comment, occurrence=occurrence_count,
                                     track_history="alert raised on " + str(datetime.datetime.today()),
                                     alert_type=alert_type)
                x.save()
                #print("reported incident!")
            elif change_permission:
                x = incidents[0]
                x.severity = severity
                x.spotted_on = spotted_on
                x.ref_seconds = ref_sec
                x.area = cam_area
                x.comments = comment
                x.occurrence = occurrence_count
                x.track_history += " \n alert altered on " + str(datetime.datetime.today())
                x.save()
                print("altered incident!")
        elif search_value == "set_success":
            change_permission = User.objects.get(
                id=request.session['_auth_user_id']).has_perm('viewer.change_IncidentReportDB')
            print("the user has: ", change_permission)
            if len(incidents) > 0 and change_permission:
                x = incidents[0]
                x.current_status = "closed"
                x.track_history += " \n alert closed on " + str(datetime.datetime.today())
                x.save()
                print("closed incident!")
    if len(incidents) > 0:
        print("yes")
        i = incidents[0]
        snap = SnapsDB.objects.filter(Q(ref_seconds=str(reference_second))).order_by('-pk')
        if len(snap) > 0:
            snap = snap[0].snap
        else:
            snap = ""
        occurrences = i.occurrence
        Area = i.area
        spotted_on = i.spotted_on
        severity = i.severity
        trackHistory = i.track_history
        comments = i.comments
        status = i.current_status
    else:
        records = AlertsDB.objects.filter(Q(ref_seconds=str(reference_second))).order_by('-pk')
        snap = ""
        Area = "none"
        spotted_on = "--"
        comments = ""
        occurrences = len(records)
        status = "unknown"
        if len(records) > 0:
            records = records[0]
            occurrences = number_of_occurences_today([Q(cam_id=records.cam_id),
                                                      Q(cam_name=records.cam_name),
                                                      Q(cam_area=records.cam_area),
                                                      Q(alert_type=records.alert_type)])
            Area = records.cam_area
            if len(Area) == 0:
                Area = records.cam_name
            D = parse(records.alert_start_time)
            if D.hour > 12:
                hh = D.hour - 12
                hr = "pm"
            else:
                hh = D.hour
                hr = "am"
            spotted_on = D.strftime('%b %d, %Y')
            spotted_on += " {hh}:{mm}:{ss} {hr}".format(hh=hh, mm=D.minute, ss=D.second, hr=hr)
            snap = SnapsDB.objects.filter(Q(ref_seconds=str(reference_second))).order_by('-pk')[0].snap
            #print({"occurrences": occurrences, "area": Area, "spotted_on": spotted_on})
            trackHistory = "None"
        severity = "Low"
        trackHistory = "None"
    #print("trackHistory ", trackHistory)
    return render(request, "viewer/incident_report.html", {"occurrences": occurrences,
                                                           "area": Area,
                                                           "snap": snap,
                                                           "spotted_on": spotted_on,
                                                           "severity": severity,
                                                           "track_history": trackHistory.strip(),
                                                           "comments": comments,
                                                           "status": status})


def change_time_format(time_format):
    if time_format == "processing" or time_format == "-":
        return time_format
    # print(time_format, type(time_format))
    D = parse(time_format)
    if D.hour > 12:
        hh = D.hour - 12
        hr = "pm"
    else:
        hh = D.hour
        hr = "am"
    return "{day}-{month}-{year} {hh}:{mm}:{ss} {hr}".format(day=D.day, month=D.month, year=D.year, hh=hh, mm=D.minute,
                                                             ss=D.second, hr=hr)


def get_alerts(n, filt, from_date=None, to_date=None, page_number=1, cam_area="all", number_of_records=20):
    reference_secs = []
    date_flag = False
    
    queries = []
    
    if from_date is not None and to_date is not None:
        date_flag = True
        print("date filter appended")
        queries.append(Q(alert_start_time__range=[from_date, to_date]))
    if filt != "view_all":
        print("alert type filter appended")
        queries.append(Q(alert_type=str(filt)))
    if cam_area != "all":
        print("cam area filter appended")
        queries.append(Q(cam_name=str(cam_area)))
    alert_get = AlertsDB.objects.all()
    for q in queries:
        alert_get = alert_get.filter(q)
    if number_of_records == -1:
        alert_get = alert_get.order_by('-pk')
        alert_get = [[w.alert_type,
                      w.cam_id,
                      w.cam_name,
                      change_time_format(w.alert_start_time),
                      change_time_format(w.alert_end_time),
                      w.cam_type] for w in alert_get]
        return alert_get
    else:
        alert_get = alert_get.order_by('-pk')[(page_number - 1) * number_of_records:page_number * number_of_records]
    images = []
    for obj in alert_get:
        ref_sec = obj.ref_seconds
        reference_secs.append(ref_sec)
        # print(ref_sec)
        obj_s = [w for w in SnapsDB.objects.filter(Q(ref_seconds=str(ref_sec)))][-1]
        images.append(obj_s.snap)
    alert_get = [[w.alert_type,
                  w.cam_id,
                  w.cam_name,
                  change_time_format(w.alert_start_time),
                  change_time_format(w.alert_end_time),
                  w.cam_type] for w in alert_get]
    return zip(images, alert_get, reference_secs)


@login_required()
def alerts_redirect(request):
    return HttpResponseRedirect("/viewer/Alerts/0")


def get_cam_areas(cam_area):
    types = []
    all_alerts = AlertsDB.objects.all()
    for alert in all_alerts:
        if alert.cam_name not in types:
            types.append(alert.cam_name)
    if cam_area == "all":
        st = '<option value="all" selected>All</option>'
    else:
        st = '<option value="all">All</option>'
    for ty in types:
        if ty == cam_area:
            st += '<option value="' + ty + '" selected >' + ty + '</option>'
        else:
            st += '<option value="' + ty + '" >' + ty + '</option>'
    return st


@login_required(login_url="/viewer/login")
def alerts(request, page_number):
    cam_area = "all"
    if page_number <= 0:
        page_number = 1
    # print("page:", page_number, type(page_number))
    global alerts_filter
    if request.method == "POST":
        export_csv = False
        dat = dict(request.POST)
        report_incidents = False
        export_pdf = False
        for k in dat.keys():
            if k.startswith("report"):
                report_incidents = k
                break
        # print(report_incidents.split('report')[-1])
        if report_incidents:
            return redirect("incident_report", report_incidents.split('report')[-1])
        print("alerts request: ", dat)
        print(dat["filter"][0])
        filt = dat["filter"][0]
        from_date = None
        if filt == "Prev":
            if page_number <= 1:
                page_number = 1
            else:
                page_number = page_number - 1
        if filt == "Next":
            page_number += 1
        if filt == "Export CSV":
            export_csv = True
        elif filt == "Export PDF":
            export_pdf = True
        print("page number: ", page_number)
        to_date = None
        if "from" in dat.keys() and "to" in dat.keys():
            if len(dat["from"][0]) > 0 and len(dat["to"][0]) > 0:
                from_date = dat["from"][0] + " 00:00:00"
                to_date = dat["to"][0] + " 23:59:59"
        if filt == "Filter" or filt == "Prev" or filt == "Next" or filt == "Export CSV" or filt == "Export PDF":
            if len(alerts_filter) > 0:
                filt = alerts_filter
            else:
                filt = "View All"
        if filt == "Crowd Gathering":
            alerts_filter = filt
            filt = "Crowd Gathering"
        elif filt == "People Count":
            alerts_filter = filt
            filt = "People Count"
        elif filt == "Sitting Posture":
            alerts_filter = filt
            filt = "Sitting Posture"
        elif filt == "View All":
            alerts_filter = filt
            filt = "view_all"
        classes = {
            "view_all": "btn",
            "Crowd": "btn",
            "People": "btn",
            "Sitting": "btn",
            "entry_exit": "btn"
        }
        if "cam_name" in dat.keys():
            cam_area = dat["cam_name"][0].lower()
        if export_csv:
            print("called to get alerts from db")
            csv_rows = get_alerts(20, filt, from_date=from_date, to_date=to_date, page_number=page_number,
                                  cam_area=cam_area, number_of_records=-1)
            field_names = ["alert_type", "cam_id", "cam_name", "start_time", "end_time", "cam_type"]
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename={}.csv'.format("download")
            writer = csv.writer(response)
            writer.writerow(field_names)
            for obj in csv_rows:
                row = writer.writerow(obj)
            return response
        elif export_pdf:
            datas = []
            print("called to get alerts from db")
            csv_rows = get_alerts(20, filt, from_date=from_date, to_date=to_date, page_number=page_number,
                                  cam_area=cam_area, number_of_records=-1)
            field_names = ["alert_type", "cam_id", "cam_name", "start_time", "end_time", "cam_type"]
            datas.append(field_names)
            for r in csv_rows:
                datas.append(r)
            response = HttpResponse(content_type='text/pdf')
            response['Content-Disposition'] = 'attachment; filename={}.pdf'.format("download")
            doc = SimpleDocTemplate(response, rightMargin=0, leftMargin=6.5, topMargin=1, bottomMargin=1)
            elements = []
            table = Table(datas)
            elements.append(table)
            doc.build(elements)
            return response
        page_alerts = get_alerts(20, filt, from_date=from_date, to_date=to_date,
                                 page_number=page_number, cam_area=cam_area)
        temp = {"alerts": page_alerts}
        classes[filt.split(' ')[0]] = "btn btn-info"
        dates = {
            "from": "",
            "to": ""
        }
        if (from_date is not None) and (to_date is not None) > 0:
            dates = {
                "from": from_date.split(" ")[0],
                "to": to_date.split(" ")[0]
            }
        temp.update(classes)
        temp.update(dates)
        temp["page_number"] = page_number
        temp["prev"] = "/viewer/Alerts/" + str(page_number - 1)
        temp["next"] = "/viewer/Alerts/" + str(page_number + 1)
        temp["options"] = get_cam_areas(cam_area)
        # print(classes)
        return render(request, "viewer/alerts.html", temp)
    else:
        classes = {
            "view_all": "btn btn-info",
            "Crowd": "btn",
            "People": "btn",
            "Sitting": "btn",
            "entry_exit": "btn"
        }
        dates = {
            "from": "",
            "to": ""
        }
        page_alerts = get_alerts(20, "view_all", page_number=page_number)
        temp = {"alerts": page_alerts}
        temp.update(classes)
        temp.update(dates)
        temp["page_number"] = page_number
        temp["prev"] = "/viewer/Alerts/" + str(page_number - 1)
        temp["next"] = "/viewer/Alerts/" + str(page_number + 1)
        temp["options"] = get_cam_areas(cam_area)
    return render(request, "viewer/alerts.html", temp)


def logout(request):
    auth.logout(request)
    return redirect('login')


def login_page(request):
    if request.method == 'POST':
        user_name = request.POST["user_name"]
        pass_word = request.POST["pass_word"]
        # print(user_name, pass_word)
        user = auth.authenticate(username=user_name, password=pass_word)
        if user is not None:
            global cap
            # cap=init_cam()
            # cv2.waitKey(5)
            print("logged in!")
            auth.login(request, user)
            return redirect("index")
        else:
            messages.info(request, "Incorrect Username or password")
            return redirect('login')
    else:
        return render(request, 'viewer/login.html')


def data_analytics(request, feature):
    print("")
    if feature == "safety":
        j = safety_stats()
        return HttpResponse(json.dumps(j), content_type='application/json')
    elif feature == "productivity":
        j = productivity_stats()
        return HttpResponse(json.dumps(j), content_type='application/json')
    elif feature == "incident_report":
        j = incident_db_list(25)
        return HttpResponse(json.dumps(j), content_type='application/json')
    
    elif feature == "alerts_by_feature":
        j = get_alerts_by_feature(5, "Crowd Gathering")
        return HttpResponse(json.dumps(j), content_type='application/json')
    
    elif feature == "alerts_by_area":
        j = get_alerts_by_area(5, "Crowd Gathering")
        return HttpResponse(json.dumps(j), content_type='application/json')
    
    elif feature == "get_alerts_by_time":
        j = get_alerts_by_time(10, "today")
        return HttpResponse(json.dumps(j), content_type='application/json')
    return HttpResponse(json.dumps([{}]), content_type="application/json")



def viewer_alertDB(request):
    types = []
    x = AlertsDB.objects.all()
    t_json = serializers.serialize('json', x)
    dump = json.dumps(t_json)
    return HttpResponse(dump, content_type='application/json')  

def userAcess(request):
    superuser =User.objects.get(id= request.session['_auth_user_id']).is_superuser
    staff =User.objects.get(id= request.session['_auth_user_id']).is_staff
    user_json = {'superuser':superuser,'is_staff':staff }
    dump = json.dumps(user_json)
    return HttpResponse(dump, content_type='application/json')        
        
def createUser(request):
    
    userData = User.objects.all()
    user_json = serializers.serialize('json', userData)
    
    dump = json.dumps(user_json)
    return HttpResponse(dump, content_type='application/json')

def user_delete(request):
    if request.method == 'GET' and request.GET['user_id'] != ' ':
        user = User.objects.get(id= request.GET['user_id'])
        user.is_active = 0
        user.save()
    return render(request, 'viewer/addUser.html')

    
def addUser(request):
    auth_user =User.objects.get(id= request.session['_auth_user_id']).is_superuser
    if auth_user == True:
        if request.method == "POST":
            user = User.objects.create_user(request.POST['userName'], request.POST['emailAddress'],request.POST['password'])
            if(request.POST['user_role'] == 'superadmin'):
                user.is_superuser = 1   
            elif(request.POST['user_role'] == 'admin'):
                user.is_staff = 1
            
            user.first_name = request.POST['firstName']
            user.last_name = request.POST['lastName']
            user.save()
        return render(request, 'viewer/addUser.html')
    else:
        return render(request, 'viewer/index.html')