# Generated by Django 4.2.6 on 2024-08-08 11:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('resume', '0005_alter_skill_level'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='link',
            field=models.URLField(blank=True, null=True),
        ),
    ]
