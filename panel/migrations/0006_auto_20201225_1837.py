# Generated by Django 3.1.4 on 2020-12-25 18:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0005_server_log'),
    ]

    operations = [
        migrations.CreateModel(
            name='Package',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('name', models.CharField(max_length=32)),
                ('size', models.IntegerField()),
                ('archive', models.FileField(upload_to='')),
            ],
        ),
        migrations.AlterField(
            model_name='server',
            name='log',
            field=models.TextField(blank=True, default=None, null=True),
        ),
        migrations.AlterField(
            model_name='server',
            name='ssh_key',
            field=models.BooleanField(verbose_name='Connect via ssh key'),
        ),
    ]