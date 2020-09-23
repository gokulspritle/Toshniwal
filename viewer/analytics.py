from .models import SnapsDB, AlertsDB, IncidentReportDB
import datetime
from django.db.models import Q


def top_priority_incidents(n):
    incidents = []
    medium_alerts = []
    low_alerts = []
    high_alerts = IncidentReportDB.objects.filter(Q(severity="High"), Q(current_status="open")).order_by("-pk")
    if len(high_alerts) < n:
        medium_alerts = IncidentReportDB.objects.filter(Q(severity="Medium"), Q(current_status="open")).order_by("-pk")
    if len(medium_alerts) < n - len(high_alerts):
        low_alerts = IncidentReportDB.objects.filter(Q(severity="Low"), Q(current_status="open")).order_by("-pk")
    incidents = high_alerts[:n]
    incidents.extend(medium_alerts[:(n - len(incidents))])
    incidents.extend(low_alerts[:(n - len(incidents))])
    print(incidents)
    results = []
    severity_map = {
        "High": 95,
        "Medium": 65,
        "Low": 25
    }
    for i in incidents:
        results.append([i.comments, i.area, severity_map[i.severity], i.ref_seconds])
    return results


def number_of_occurences_today(filters):
    today = datetime.datetime.today()
    curr_st = str(today).split(" ")[0] + " 00:00:00"
    curr_end = str(today)
    alerts = AlertsDB.objects.filter(Q(alert_start_time__range=[curr_st, curr_end]))
    for f in filters:
        alerts = alerts.filter(f)
    number_of_occurrences = len(alerts)
    return number_of_occurrences


def top_alerts_today():
    today = datetime.datetime.today()
    curr_st = str(today).split(" ")[0] + " 00:00:00"
    curr_end = str(today)
    crowd_alerts_today = AlertsDB.objects.filter(Q(alert_start_time__range=[curr_st, curr_end]))
    if crowd_alerts_today == 1:
        return "first", ""
    if crowd_alerts_today == 2:
        return "second", ""
    if crowd_alerts_today == 3:
        return "third", ""
    if crowd_alerts_today > 3:
        return str(crowd_alerts_today) + "th", "\nyour productivity might decrease"
    return str("first"), ""


def safety_stats():
    # people count
    today = datetime.datetime.today()
    result_list = []
    day_map = {
        -9:5,
        -8:5,
        -7:8,
        -6:10,
        -5:15,
        -4:20,
        -3:25,
        -2:30,
        -1:35
    }
    dummy_before = datetime.datetime(2020, 9, 9)
    past_seven_days = [today - datetime.timedelta(days=i) for i in range(7)]
    for curr in past_seven_days:
        curr_st = str(curr).split(" ")[0] + " 00:00:00"
        curr_end = str(curr).split(" ")[0] + " 23:59:59"
        crowd_alerts_today = AlertsDB.objects.filter(Q(alert_start_time__range=[curr_st, curr_end]),
                                                     Q(alert_type="People Count"))
        count = len(crowd_alerts_today)
        if curr < dummy_before:
            count = day_map[(curr-dummy_before).days]
            
        result_list.append({"day": curr.strftime("%B")[:3]+", "+curr.strftime("%d")+"-"+curr.strftime("%a"), "count": count})
    return result_list

def productivity_stats():
    # Crowd Gathering, Sitting Posture
    today = datetime.datetime.today()
    result_list = []
    day_map = {

        -9: 5,
        -8: 8,
        -7: 10,
        -6: 15,
        -5: 20,
        -4: 22,
        -3: 25,
        -2: 28,
        -1: 30
    }
    dummy_before = datetime.datetime(2020, 9, 10)
    past_seven_days = [today - datetime.timedelta(days=i) for i in range(7)]
    for curr in past_seven_days:
        curr_st = str(curr).split(" ")[0] + " 00:00:00"
        curr_end = str(curr).split(" ")[0] + " 23:59:59"
        crowd_alerts_today = AlertsDB.objects.filter(Q(alert_start_time__range=[curr_st, curr_end]),
                                                     (Q(alert_type="Crowd Gathering") | Q(
                                                         alert_type="Sitting Posture")))
        count = len(crowd_alerts_today)
        if curr < dummy_before:
            count = day_map[(curr - dummy_before).days]
        result_list.append({"day": curr.strftime("%B")[:3] + ", " + curr.strftime("%d")+"-"+curr.strftime("%a"), "count": count})
    return result_list


def incident_db_list(n):
    # this returns {"comments": i.comments,
    #                         "location": i.area,
    #                         "alert_level": severity_map[i.severity],
    #                         "t_id": i.ref_seconds,
    #                         "event_type": i.alert_type,
    #                         "time":i.spotted_on}
    # Action center can just be viewer/incident_report/t_id [ example: http://localhost:8000/viewer/incident_report/1599033052.7589333 ]
    incidents = []
    medium_alerts = []
    low_alerts = []
    high_alerts = IncidentReportDB.objects.filter(Q(severity="High"), Q(current_status="open")).order_by("-pk")
    if len(high_alerts) < n:
        medium_alerts = IncidentReportDB.objects.filter(Q(severity="Medium"), Q(current_status="open")).order_by("-pk")
    if len(medium_alerts) < n - len(high_alerts):
        low_alerts = IncidentReportDB.objects.filter(Q(severity="Low"), Q(current_status="open")).order_by("-pk")
    incidents = high_alerts[:n]
    incidents.extend(medium_alerts[:(n - len(incidents))])
    incidents.extend(low_alerts[:(n - len(incidents))])
    results = []
    severity_map = {
        "High": 95,
        "Medium": 65,
        "Low": 25
    }
    for i in incidents:
        results.append({"comments": i.comments,
                        "location": i.area,
                        "alert_level": severity_map[i.severity],
                        "t_id": i.ref_seconds,
                        "event_type": i.alert_type,
                        "time": i.spotted_on})
    return results


def get_alerts_by_feature(n, alert_type):
    # alert type can be, People Counting, Crowd Gathering, Sitting Posture [ Note: Case Sensitive ]
    # this returns {"comments": i.comments,
    #                         "location": i.area,
    #                         "alert_level": severity_map[i.severity],
    #                         "t_id": i.ref_seconds,
    #                         "event_type": i.alert_type,
    #                         "time":i.spotted_on}
    # Action center can just be viewer/incident_report/t_id [ example: http://localhost:8000/viewer/incident_report/1599033052.7589333 ]
    incidents = []
    medium_alerts = []
    low_alerts = []
    high_alerts = IncidentReportDB.objects.filter(Q(severity="High"), Q(current_status="open"),
                                                  Q(alert_type=alert_type)).order_by("-pk")
    if len(high_alerts) < n:
        medium_alerts = IncidentReportDB.objects.filter(Q(severity="Medium"), Q(current_status="open"),
                                                        Q(alert_type=alert_type)).order_by("-pk")
    if len(medium_alerts) < n - len(high_alerts):
        low_alerts = IncidentReportDB.objects.filter(Q(severity="Low"), Q(current_status="open"),
                                                     Q(alert_type=alert_type)).order_by("-pk")
    incidents = high_alerts[:n]
    incidents.extend(medium_alerts[:(n - len(incidents))])
    incidents.extend(low_alerts[:(n - len(incidents))])
    print(incidents)
    results = []
    severity_map = {
        "High": 95,
        "Medium": 65,
        "Low": 25
    }
    for i in incidents:
        results.append({"comments": i.comments,
                        "location": i.area,
                        "alert_level": severity_map[i.severity],
                        "t_id": i.ref_seconds,
                        "event_type": i.alert_type,
                        "time": i.spotted_on})
    return results


def get_alerts_by_area(n, area):
    # alert type can be, People Counting, Crowd Gathering, Sitting Posture [ Note: Case Sensitive ]
    # this returns {"comments": i.comments,
    #                         "location": i.area,
    #                         "alert_level": severity_map[i.severity],
    #                         "t_id": i.ref_seconds,
    #                         "event_type": i.alert_type,
    #                         "time":i.spotted_on}
    # Action center can just be viewer/incident_report/t_id [ example: http://localhost:8000/viewer/incident_report/1599033052.7589333 ]
    incidents = []
    medium_alerts = []
    low_alerts = []
    high_alerts = IncidentReportDB.objects.filter(Q(severity="High"), Q(current_status="open"),
                                                  Q(area=area)).order_by("-pk")
    if len(high_alerts) < n:
        medium_alerts = IncidentReportDB.objects.filter(Q(severity="Medium"), Q(current_status="open"),
                                                        Q(area=area)).order_by("-pk")
    if len(medium_alerts) < n - len(high_alerts):
        low_alerts = IncidentReportDB.objects.filter(Q(severity="Low"), Q(current_status="open"),
                                                     Q(area=area)).order_by("-pk")
    incidents = high_alerts[:n]
    incidents.extend(medium_alerts[:(n - len(incidents))])
    incidents.extend(low_alerts[:(n - len(incidents))])
    print(incidents)
    results = []
    severity_map = {
        "High": 95,
        "Medium": 65,
        "Low": 25
    }
    for i in incidents:
        results.append({"comments": i.comments,
                        "location": i.area,
                        "alert_level": severity_map[i.severity],
                        "t_id": i.ref_seconds,
                        "event_type": i.alert_type,
                        "time": i.spotted_on})
    return results


def get_alerts_by_time(n, filter):
    today = datetime.datetime.today()
    incidents = []
    if filter == "month":
        incidents = IncidentReportDB.filter(spotted_on__year=today.year, spotted_on__month=today.month)
    elif filter == "today":
        incidents = IncidentReportDB.filter(spotted_on=today)
    elif filter == "week":
        start = datetime.datetime.today() - datetime.timedelta(days=7)
        start = str(start).split(" ")[0] + " 00:00:00"
        incidents = AlertsDB.objects.filter(Q(alert_start_time__range=[start, today]))
    results = []
    severity_map = {
        "High": 95,
        "Medium": 65,
        "Low": 25
    }
    for i in incidents:
        results.append({"comments": i.comments,
                        "location": i.area,
                        "alert_level": severity_map[i.severity],
                        "t_id": i.ref_seconds,
                        "event_type": i.alert_type,
                        "time": i.spotted_on})
    return results


