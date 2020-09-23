# Generated by Django 3.0.7 on 2020-07-13 11:54

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('viewer', '0002_auto_20200713_1722'),
    ]

    operations = [
        migrations.CreateModel(
            name='AlertsDB',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ref_seconds', models.CharField(max_length=40)),
                ('alert_type', models.CharField(max_length=20)),
                ('alert_start_time', models.CharField(max_length=40)),
                ('alert_end_time', models.CharField(max_length=40)),
                ('cam_id', models.CharField(max_length=20)),
                ('cam_name', models.CharField(max_length=30)),
                ('cam_type', models.CharField(max_length=8)),
            ],
        ),
        migrations.CreateModel(
            name='SnapsDB',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('snap', models.CharField(max_length=80000)),
                ('ref_seconds', models.CharField(max_length=40)),
            ],
        ),
    ]
