from django.db import models
import time


class AlertsDB(models.Model):
    ref_seconds = models.CharField(max_length=40)
    alert_type = models.CharField(max_length=20)
    alert_start_time = models.CharField(max_length=40)
    alert_end_time = models.CharField(max_length=40)
    cam_id = models.CharField(max_length=20)
    cam_name = models.CharField(max_length=30)
    cam_type = models.CharField(max_length=8)
    cam_area = models.CharField(max_length=100)


    def __str__(self):
        return "(%r, %r, %r,%r, %r, %r)" %(self.ref_seconds,self.alert_type,self.cam_id,self.cam_area,self.cam_name,self.alert_end_time)


class SnapsDB(models.Model):
    snap = models.CharField(max_length=300000)
    ref_seconds = models.CharField(max_length=40)

    def __str__(self):
        return self.ref_seconds


class CamDB(models.Model):
    cam_address = models.CharField(max_length=100)
    cam_type = models.CharField(max_length=100)
    cam_name = models.CharField(max_length=100)
    cam_id = models.CharField(max_length=100)
    cam_action = models.CharField(max_length=100)
    cam_area = models.CharField(max_length=100)
    group = models.CharField(max_length=100)


    def __str__(self):
        return self.cam_address, self.cam_type, self.cam_name, self.cam_id, self.cam_action, self.group, self.cam_area


class CamTransformationMatrix(models.Model):
    cam_address = models.CharField(max_length=300)
    transformationMatrix = models.CharField(max_length=1000)
    distanceThresh = models.CharField(max_length=10)

    def __str__(self):
        return self.cam_address + str(self.transformationMatrix)


class IncidentReportDB(models.Model):
    ref_seconds = models.CharField(max_length=40)
    spotted_on = models.CharField(max_length=200)
    area = models.CharField(max_length=20)
    occurrence = models.CharField(max_length=6)
    severity = models.CharField(max_length=10)
    track_history = models.CharField(max_length=1000)
    comments = models.CharField(max_length=800)
    current_status = models.CharField(max_length=50, default="open")
    alert_type = models.CharField(max_length=20, default="placeholder")

    def __str__(self):
        return self.spotted_on + self.comments + self.track_history
