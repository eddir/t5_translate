# Generated by Django 3.1.7 on 2021-03-21 13:45

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0016_dedic_condition'),
    ]

    operations = [
        migrations.AddField(
            model_name='dedic',
            name='last_listen',
            field=models.DateTimeField(default=datetime.datetime(2021, 3, 21, 13, 45, 8, 772846)),
            preserve_default=False,
        ),
    ]