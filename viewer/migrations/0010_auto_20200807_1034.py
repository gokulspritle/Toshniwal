# Generated by Django 3.0.7 on 2020-08-07 05:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0009_auto_20200721_2034'),
    ]

    operations = [
        migrations.AddField(
            model_name='alertsdb',
            name='cam_area',
            field=models.CharField(default='default', max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='camdb',
            name='cam_area',
            field=models.CharField(default='def', max_length=100),
            preserve_default=False,
        ),
    ]
