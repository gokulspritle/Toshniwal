# Generated by Django 3.0.7 on 2020-09-05 10:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0016_incidentreportdb_current_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='incidentreportdb',
            name='alert_type',
            field=models.CharField(default='placeholder', max_length=20),
        ),
    ]
