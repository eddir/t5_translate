# Generated by Django 3.2.6 on 2021-09-08 14:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0026_auto_20210907_1812'),
    ]

    operations = [
        migrations.AddField(
            model_name='server',
            name='processes',
            field=models.TextField(blank=True, default=None, null=True),
        ),
    ]