# Generated by Django 3.1.4 on 2020-12-28 20:42

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('panel', '0008_auto_20201228_1602'),
    ]

    operations = [
        migrations.AddField(
            model_name='server',
            name='m_package',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.PROTECT, to='panel.mpackage'),
        ),
        migrations.AddField(
            model_name='server',
            name='sr_package',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.PROTECT, to='panel.srpackage'),
        ),
    ]