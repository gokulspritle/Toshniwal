# Generated by Django 3.0.7 on 2020-07-13 11:52

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0001_initial'),
    ]

    operations = [
        migrations.DeleteModel(
            name='AlertsDB',
        ),
        migrations.DeleteModel(
            name='SnapsDB',
        ),
    ]