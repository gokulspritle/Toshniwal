# Generated by Django 3.0.7 on 2020-07-21 15:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0006_camdb'),
    ]

    operations = [
        migrations.RenameField(
            model_name='camdb',
            old_name='cam_function',
            new_name='cam_action',
        ),
    ]
