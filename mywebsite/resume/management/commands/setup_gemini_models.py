from django.core.management.base import BaseCommand
from django.conf import settings
from resume.models import GeminiModel  # your_app'i kendi app adınızla değiştirin


class Command(BaseCommand):
    help = 'Gemini modellerini otomatik olarak oluşturur'

    def add_arguments(self, parser):
        parser.add_argument(
            '--api-key',
            type=str,
            help='Gemini API anahtarı',
        )
        parser.add_argument(
            '--default-model',
            type=str,
            default='gemini-2.0-flash-exp',
            choices=[
                'gemini-2.0-flash-exp',
                'gemini-2.5-pro',
                'gemini-1.5-pro',
                'gemini-1.5-flash',
                'gemini-pro'
            ],
            help='Varsayılan olarak aktif edilecek model',
        )

    def handle(self, *args, **options):
        api_key = options.get('api_key')
        default_model = options.get('default_model')

        if not api_key:
            # settings.py'den API key almaya çalış
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not api_key:
                self.stdout.write(
                    self.style.ERROR(
                        'API anahtarı gerekli! --api-key parametresi ile veya settings.GEMINI_API_KEY ile sağlayın.')
                )
                return

        # Model tanımları
        models_to_create = [
            {
                'name': 'Gemini 2.0 Flash (Experimental)',
                'model_identifier': 'gemini-2.0-flash-exp',
                'description': 'En yeni deneysel Gemini modeli. Hızlı ve güçlü.',
                'max_tokens': 2048,
                'temperature': 0.7
            },
            {
                'name': 'Gemini 2.5 Pro',
                'model_identifier': 'gemini-2.5-pro',
                'description': 'Gemini 2.5 Pro modeli. Yüksek performans.',
                'max_tokens': 2048,
                'temperature': 0.7
            },
            {
                'name': 'Gemini 1.5 Pro',
                'model_identifier': 'gemini-1.5-pro',
                'description': 'Gemini 1.5 Pro modeli. Dengeli performans.',
                'max_tokens': 2048,
                'temperature': 0.7
            },
            {
                'name': 'Gemini 1.5 Flash',
                'model_identifier': 'gemini-1.5-flash',
                'description': 'Gemini 1.5 Flash modeli. Hızlı yanıtlar.',
                'max_tokens': 2048,
                'temperature': 0.7
            },
            {
                'name': 'Gemini Pro',
                'model_identifier': 'gemini-pro',
                'description': 'Klasik Gemini Pro modeli.',
                'max_tokens': 8192,
                'temperature': 0.7
            }
        ]

        created_count = 0
        updated_count = 0

        for model_data in models_to_create:
            model, created = GeminiModel.objects.get_or_create(
                model_identifier=model_data['model_identifier'],
                defaults={
                    'name': model_data['name'],
                    'api_key': api_key,
                    'description': model_data['description'],
                    'max_tokens': model_data['max_tokens'],
                    'temperature': model_data['temperature'],
                    'is_enabled': True,
                    'is_active': model_data['model_identifier'] == default_model,
                    'is_default': model_data['model_identifier'] == default_model
                }
            )

            if created:
                created_count += 1
                self.stdout.write(
                    self.style.SUCCESS(f'Model oluşturuldu: {model.name}')
                )
            else:
                # Mevcut modeli güncelle
                model.api_key = api_key
                if model_data['model_identifier'] == default_model:
                    # Diğer modelleri pasif et
                    GeminiModel.objects.exclude(pk=model.pk).update(
                        is_active=False, is_default=False
                    )
                    model.is_active = True
                    model.is_default = True
                model.save()
                updated_count += 1
                self.stdout.write(
                    self.style.WARNING(f'Model güncellendi: {model.name}')
                )

        self.stdout.write('\n')
        self.stdout.write(self.style.SUCCESS(f'İşlem tamamlandı!'))
        self.stdout.write(f'Oluşturulan model sayısı: {created_count}')
        self.stdout.write(f'Güncellenen model sayısı: {updated_count}')

        # Aktif modeli göster
        try:
            active_model = GeminiModel.get_active_model()
            self.stdout.write(f'Aktif model: {active_model.name}')
        except:
            self.stdout.write(self.style.WARNING('Aktif model bulunamadı!'))

        self.stdout.write('\n')
        self.stdout.write('Kullanım:')
        self.stdout.write('Admin panelden /admin/your_app/geminimodel/ adresine gidip modelleri yönetebilirsiniz.')
        self.stdout.write('Model değiştirmek için ilgili modelin "Aktif Et" butonuna tıklayın.')