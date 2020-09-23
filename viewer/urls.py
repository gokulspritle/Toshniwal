from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('/', views.index_new, name='index_new'),
    path('dashboard', views.dashboard, name='dashboard'),
    path('stream', views.stream_cam, name="stream"),
    path('update_data', views.update_data, name='update_data'),
    path('getData', views.get_data, name="get_data"),
    path('update_restrict_data', views.update_restrict_data, name="update_restrict_data"),
    path('CamSettings', views.cam_settings, name="camera_settings"),
    path('createUser', views.createUser, name="createUser"),
    path('user_delete', views.user_delete, name="user_delete"),
    path('userAcess', views.userAcess, name="userAcess"),
    path('Alerts/<int:page_number>', views.alerts, name="Alerts"),
    path('Alerts', views.alerts_redirect, name="Alerts_redirect"),
    path('login', views.login_page, name="login"),
    path('logout', views.logout, name="logout"),
    path('analytics', views.analytics, name="analytics"),
    path('stream_new/<int:stream_number>', views.stream_cam_multiple, name="stream"),
    path('incident_report/<str:reference_second>', views.report_incident, name="incident_report"),
    path('management', views.management_dashboard, name="management"),
    path('db_analytics/<str:feature>', views.data_analytics, name="data_analytics"),
    path('viewer_alertDB', views.viewer_alertDB, name="viewer_alertDB"),
    path('addUser', views.addUser, name="addUser"),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
