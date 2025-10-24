from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta
from resume.models import ContactMessage


class Command(BaseCommand):
    help = 'Eski iletişim mesajlarını temizler'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=365,
            help='Kaç gün önceki mesajları sil (varsayılan: 365)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Gerçek silme işlemi yapmadan test et'
        )

    def handle(self, *args, **options):
        days = options['days']
        dry_run = options['dry_run']

        cutoff_date = timezone.now() - timedelta(days=days)

        old_messages = ContactMessage.objects.filter(
            created_at__lt=cutoff_date
        )

        count = old_messages.count()

        if dry_run:
            self.stdout.write(
                self.style.WARNING(f'{count} mesaj silinecek (dry-run mode)')
            )
        else:
            old_messages.delete()
            self.stdout.write(
                self.style.SUCCESS(f'{count} eski mesaj silindi')
            )