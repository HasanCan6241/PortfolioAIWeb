# Generated by Django 4.2.6 on 2024-08-08 11:16

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resume', '0004_about'),
    ]

    operations = [
        migrations.AlterField(
            model_name='skill',
            name='level',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]
