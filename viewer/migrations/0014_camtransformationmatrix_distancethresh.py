# Generated by Django 3.0.7 on 2020-08-11 12:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('viewer', '0013_camtransformationmatrix'),
    ]

    operations = [
        migrations.AddField(
            model_name='camtransformationmatrix',
            name='distanceThresh',
            field=models.CharField(default=0, max_length=10),
            preserve_default=False,
        ),
    ]
