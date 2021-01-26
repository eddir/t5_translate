# Generated by Django 3.1.4 on 2020-12-05 18:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='server',
            name='ip',
            field=models.GenericIPAddressField(verbose_name='Ip address'),
        ),
        migrations.AlterField(
            model_name='server',
            name='name',
            field=models.CharField(max_length=32, verbose_name='Server name'),
        ),
        migrations.AlterField(
            model_name='server',
            name='password_single',
            field=models.CharField(blank=True, max_length=32),
        ),
    ]