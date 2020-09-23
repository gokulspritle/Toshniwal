# Generated by Django 3.0.7 on 2020-07-21 14:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0005_auto_20200713_1812'),
    ]

    operations = [
        migrations.CreateModel(
            name='CamDB',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cam_url', models.CharField(max_length=100)),
                ('cam_type', models.CharField(max_length=100)),
                ('cam_name', models.CharField(max_length=100)),
                ('cam_id', models.CharField(max_length=100)),
                ('cam_function', models.CharField(max_length=100)),
                ('cam_group', models.CharField(max_length=100)),
            ],
        ),
    ]
