from django.contrib import admin
from .models import SnapsDB, AlertsDB, CamTransformationMatrix, CamDB, IncidentReportDB

admin.site.register(SnapsDB)
admin.site.register(AlertsDB)
admin.site.register(CamTransformationMatrix)
admin.site.register(CamDB)
admin.site.register(IncidentReportDB)
