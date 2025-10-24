from django.db import models
from cryptography.fernet import Fernet
from django.conf import settings
import os
from django.core.exceptions import ValidationError
from django.core.validators import MinValueValidator, MaxValueValidator


class Profile(models.Model):
    """Kişisel profil bilgileri"""
    profile = models.ImageField(upload_to='profile/', blank=True, null=True)
    name = models.CharField(max_length=100, default="")
    title = models.CharField(max_length=150, blank=True, null=True)
    location = models.CharField(max_length=100, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Profil"
        verbose_name_plural = "Profiller"

    def __str__(self):
        return f"Profil - {self.name}" if self.name else "Profil Resmi"


class EmailSettings(models.Model):
    """E-posta ayarları ve uygulama şifresi yönetimi"""
    name = models.CharField(max_length=100, default="Gmail Ayarları", help_text="Ayar seti adı")
    email = models.EmailField(help_text="Gmail adresi")
    encrypted_password = models.TextField(help_text="Şifrelenmiş uygulama şifresi")
    smtp_server = models.CharField(max_length=100, default="smtp.gmail.com")
    smtp_port = models.IntegerField(default=587)
    use_tls = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True, help_text="Bu ayarlar aktif mi?")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "E-posta Ayarları"
        verbose_name_plural = "E-posta Ayarları"

    def __str__(self):
        return f"{self.name} - {self.email}"

    def set_password(self, raw_password):
        """Şifreyi şifreleyen method"""
        if not raw_password:
            return
        key = self._get_encryption_key()
        f = Fernet(key)
        self.encrypted_password = f.encrypt(raw_password.encode()).decode()

    def get_password(self):
        """Şifreyi çözen method"""
        if not self.encrypted_password:
            return None
        try:
            key = self._get_encryption_key()
            f = Fernet(key)
            return f.decrypt(self.encrypted_password.encode()).decode()
        except Exception as e:
            print(f"Şifre çözme hatası: {e}")
            return None

    def _get_encryption_key(self):
        """Şifreleme anahtarını getirir"""
        key = getattr(settings, 'EMAIL_ENCRYPTION_KEY', None)
        if not key:
            # Geliştirme ortamı için otomatik anahtar üretimi
            if settings.DEBUG:
                key = Fernet.generate_key()
                print(f"UYARI: Yeni şifreleme anahtarı oluşturuldu.")
                print(f"Bunu settings.py'a ekleyin: EMAIL_ENCRYPTION_KEY = b'{key.decode()}'")
            else:
                raise ValueError("EMAIL_ENCRYPTION_KEY settings.py'de tanımlanmalı!")

        # String ise bytes'a çevir
        if isinstance(key, str):
            key = key.encode()
        return key

    @classmethod
    def get_active_settings(cls):
        """Aktif e-posta ayarlarını getirir"""
        return cls.objects.filter(is_active=True).first()

    @property
    def has_password(self):
        """Şifre tanımlı mı kontrol eder"""
        return bool(self.encrypted_password)


class ContactMessage(models.Model):
    """İletişim formu mesajları"""
    STATUS_CHOICES = [
        ('new', 'Yeni'),
        ('read', 'Okundu'),
        ('replied', 'Yanıtlandı'),
        ('archived', 'Arşivlendi'),
    ]

    name = models.CharField(max_length=100, verbose_name="Ad Soyad")
    email = models.EmailField(verbose_name="E-posta")
    subject = models.CharField(max_length=200, verbose_name="Konu")
    message = models.TextField(verbose_name="Mesaj")
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='new', verbose_name="Durum")
    ip_address = models.GenericIPAddressField(blank=True, null=True, verbose_name="IP Adresi")
    user_agent = models.TextField(blank=True, null=True, verbose_name="Tarayıcı Bilgisi")
    is_spam = models.BooleanField(default=False, verbose_name="Spam mı?")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Oluşturma Tarihi")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Güncelleme Tarihi")

    class Meta:
        verbose_name = "İletişim Mesajı"
        verbose_name_plural = "İletişim Mesajları"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.subject} ({self.get_status_display()})"

    def mark_as_read(self):
        """Mesajı okundu olarak işaretle"""
        if self.status == 'new':
            self.status = 'read'
            self.save(update_fields=['status', 'updated_at'])

    def mark_as_replied(self):
        """Mesajı yanıtlandı olarak işaretle"""
        self.status = 'replied'
        self.save(update_fields=['status', 'updated_at'])


class Education(models.Model):
    """Eğitim geçmişi"""
    school = models.CharField(max_length=100)  # Karakter limiti artırıldı
    logo = models.ImageField(upload_to='school_logos/', blank=True, null=True)
    title = models.CharField(max_length=150)
    description = models.TextField()
    gpa = models.CharField(max_length=50, blank=True, null=True)
    start_date = models.DateField(blank=True, null=True)
    end_date = models.DateField(blank=True, null=True)
    is_current = models.BooleanField(default=False)  # Halen devam ediyor mu?
    order = models.IntegerField(default=0)  # Sıralama için

    class Meta:
        verbose_name = "Eğitim"
        verbose_name_plural = "Eğitimler"
        ordering = ['-start_date']

    def __str__(self):
        return f"{self.school} - {self.title}"


class About(models.Model):
    """Hakkında bilgileri"""
    about = models.TextField()
    short_description = models.TextField(max_length=500, blank=True, null=True)  # Kısa açıklama
    years_of_experience = models.IntegerField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Hakkında"
        verbose_name_plural = "Hakkında"

    def __str__(self):
        return "Hakkında Bilgileri"


class Service(models.Model):
    """Sunulan hizmetler"""
    name = models.CharField(max_length=150)
    description = models.TextField()
    icon = models.CharField(max_length=50, blank=True, null=True)  # FontAwesome icon class
    image = models.ImageField(upload_to='services/', blank=True, null=True)
    is_active = models.BooleanField(default=True)
    order = models.IntegerField(default=0)

    class Meta:
        verbose_name = "Hizmet"
        verbose_name_plural = "Hizmetler"
        ordering = ['order']

    def __str__(self):
        return self.name


class SkillCategory(models.Model):
    """Yetenek kategorileri"""
    name = models.CharField(max_length=100)
    order = models.IntegerField(default=0)

    class Meta:
        verbose_name = "Yetenek Kategorisi"
        verbose_name_plural = "Yetenek Kategorileri"
        ordering = ['order']

    def __str__(self):
        return self.name


class Skill(models.Model):
    """Skills"""
    LEVEL_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
        ('expert', 'Expert'),
    ]

    name = models.CharField(max_length=100)
    category = models.ForeignKey(SkillCategory, on_delete=models.CASCADE, related_name='skills')
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES, blank=True, null=True)
    percentage = models.IntegerField(blank=True, null=True, help_text="0-100 arası değer")
    icon = models.CharField(max_length=50, blank=True, null=True)
    order = models.IntegerField(default=0)

    class Meta:
        verbose_name = "Yetenek"
        verbose_name_plural = "Yetenekler"
        ordering = ['category', 'order']

    def __str__(self):
        return f"{self.category.name} - {self.name}"


class ProjectCategory(models.Model):
    """Proje kategorileri"""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)  # Açıklama eklemek istersen
    icon_class = models.CharField(max_length=50, blank=True, null=True)  # FontAwesome class
    order = models.IntegerField(default=0)

    class Meta:
        verbose_name = "Proje Kategorisi"
        verbose_name_plural = "Proje Kategorileri"
        ordering = ['order']

    def __str__(self):
        return self.name



class Project(models.Model):
    """Projects"""
    STATUS_CHOICES = [
        ('completed', 'Completed'),
        ('in_progress', 'In Progress'),
        ('planned', 'Planned'),
    ]


    title = models.CharField(max_length=150)
    description = models.TextField()
    short_description = models.TextField(max_length=300, blank=True, null=True)
    category = models.ForeignKey(ProjectCategory, on_delete=models.CASCADE, related_name='projects')
    image = models.ImageField(upload_to='projects/', blank=True, null=True)
    link = models.URLField(blank=True, null=True)
    github_link = models.URLField(blank=True, null=True)
    technologies = models.ManyToManyField(Skill, blank=True, related_name='used_in_projects')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='completed')
    start_date = models.DateField(blank=True, null=True)
    end_date = models.DateField(blank=True, null=True)
    is_featured = models.BooleanField(default=False)
    order = models.IntegerField(default=0)

    class Meta:
        verbose_name = "Proje"
        verbose_name_plural = "Projeler"
        ordering = ['-start_date', 'order']

    def __str__(self):
        return self.title


class Certification(models.Model):
    """Sertifikalar"""
    title = models.CharField(max_length=150)
    institution = models.CharField(max_length=150)
    date = models.DateField()
    expiry_date = models.DateField(blank=True, null=True)  # Geçerlilik süresi
    certificate_link = models.URLField(max_length=500, blank=True, null=True)
    certificate_image = models.ImageField(upload_to='certifications/', blank=True, null=True)
    credential_id = models.CharField(max_length=100, blank=True, null=True)
    is_active = models.BooleanField(default=True)

    # Manuel sıralama için
    order = models.IntegerField(default=0)

    class Meta:
        verbose_name = "Sertifika"
        verbose_name_plural = "Sertifikalar"
        ordering = ['order', '-date']   # Önce order, sonra tarih

    def __str__(self):
        return f"{self.title} - {self.institution}"



class Experience(models.Model):
    """Work Experiences"""
    EMPLOYMENT_TYPE_CHOICES = [
        ('full_time', 'Full Time'),
        ('part_time', 'Part Time'),
        ('contract', 'Contract'),
        ('freelance', 'Freelance'),
        ('internship', 'Internship'),
    ]


    job_title = models.CharField(max_length=150)
    company = models.CharField(max_length=150)
    company_logo = models.ImageField(upload_to='companies/', blank=True, null=True)
    location = models.CharField(max_length=100, blank=True, null=True)
    employment_type = models.CharField(max_length=20, choices=EMPLOYMENT_TYPE_CHOICES, blank=True, null=True)
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    is_current = models.BooleanField(default=False)
    description = models.TextField()
    achievements = models.TextField(blank=True, null=True)  # Başarılar
    technologies = models.ManyToManyField(Skill, blank=True, related_name='used_in_experiences')

    class Meta:
        verbose_name = "Deneyim"
        verbose_name_plural = "Deneyimler"
        ordering = ['-start_date']

    def __str__(self):
        return f"{self.job_title} - {self.company}"


class Achievement(models.Model):
    """Başarılar/Ödüller"""
    logo = models.ImageField(upload_to='achievements/', blank=True, null=True)
    title = models.CharField(max_length=150)
    description = models.TextField()
    date = models.DateField(blank=True, null=True)
    organization = models.CharField(max_length=100, blank=True, null=True)
    link = models.URLField(blank=True, null=True)
    is_featured = models.BooleanField(default=False)

    # DÜZELTME: FileField kullanın, ImageField değil!
    certificate = models.FileField(
        upload_to='achievements/certificates/',
        blank=True,
        null=True,
        help_text="Sertifika veya belge dosyası (PDF, JPG, PNG, DOCX vb.)"
    )

    class Meta:
        verbose_name = "Başarı"
        verbose_name_plural = "Başarılar"
        ordering = ['-date']

    def __str__(self):
        return self.title

    @property
    def certificate_extension(self):
        """Dosya uzantısını döndürür"""
        if self.certificate:
            return os.path.splitext(self.certificate.name)[1].lower()
        return None

    @property
    def is_image_certificate(self):
        """Sertifikanın görsel dosya olup olmadığını kontrol eder"""
        if not self.certificate:
            return False
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        return self.certificate_extension in image_extensions

    @property
    def is_pdf_certificate(self):
        """Sertifikanın PDF olup olmadığını kontrol eder"""
        if not self.certificate:
            return False
        return self.certificate_extension == '.pdf'

    @property
    def certificate_icon(self):
        """Dosya türüne göre ikon döndürür"""
        if not self.certificate:
            return None

        ext = self.certificate_extension

        if ext in ['.pdf']:
            return 'fas fa-file-pdf'
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return 'fas fa-file-image'
        elif ext in ['.doc', '.docx']:
            return 'fas fa-file-word'
        elif ext in ['.xls', '.xlsx']:
            return 'fas fa-file-excel'
        elif ext in ['.ppt', '.pptx']:
            return 'fas fa-file-powerpoint'
        elif ext in ['.txt']:
            return 'fas fa-file-alt'
        else:
            return 'fas fa-file'

class Communication(models.Model):
    """Contact Information"""
    COMMUNICATION_TYPE_CHOICES = [
        ('email', 'Email'),
        ('phone', 'Phone'),
        ('linkedin', 'LinkedIn'),
        ('github', 'GitHub'),
        ('twitter', 'Twitter'),
        ('instagram', 'Instagram'),
        ('website', 'Website'),
        ('other', 'Other'),
    ]

    type = models.CharField(max_length=20, choices=COMMUNICATION_TYPE_CHOICES)
    title = models.CharField(max_length=100)
    value = models.CharField(max_length=200)  # Email, telefon numarası vs.
    communication_link = models.URLField(max_length=500, blank=True, null=True)
    icon = models.CharField(max_length=50, blank=True, null=True)
    is_public = models.BooleanField(default=True)
    order = models.IntegerField(default=0)

    class Meta:
        verbose_name = "İletişim"
        verbose_name_plural = "İletişim Bilgileri"
        ordering = ['order']

    def __str__(self):
        return f"{self.get_type_display()} - {self.title}"


class Resume(models.Model):
    """Özgeçmiş dosyaları"""
    title = models.CharField(max_length=100, default="Özgeçmiş")
    file = models.FileField(upload_to='resumes/')
    language = models.CharField(max_length=10, default='tr', help_text="tr, en, vs.")
    is_active = models.BooleanField(default=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Özgeçmiş"
        verbose_name_plural = "Özgeçmişler"
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.title} ({self.language.upper()})"


class FreelanceService(models.Model):
    name = models.CharField(max_length=100, verbose_name="Hizmet Adı")
    description = models.TextField(verbose_name="Hizmet Açıklaması")
    service_image = models.ImageField(
        upload_to='services/images/',
        blank=True,
        null=True,
        verbose_name="Hizmet Görseli",
        help_text="Hizmeti tanımlayan görsel yükleyin"
    )
    service_link = models.URLField(blank=True, null=True, verbose_name="Hizmet Linki",
                                   help_text="Bionluk, Upwork vb. hizmet linki")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Freelancer Hizmet"
        verbose_name_plural = "Freelancer Hizmetler"

    def __str__(self):
        return self.name


class Platform(models.Model):
    name = models.CharField(max_length=50, verbose_name="Platform Adı")
    platform_logo = models.ImageField(
        upload_to='platforms/logos/',
        blank=True,
        null=True,
        verbose_name="Platform Logosu",
        help_text="Platform logosunu yükleyin"
    )
    color = models.CharField(max_length=7, default="#667eea", verbose_name="Platform Rengi")
    link = models.URLField(
        max_length=200,
        blank=True,
        null=True,
        verbose_name="Platform Linki",
        help_text="Platformun web sitesi veya ilgili link"
    )

    class Meta:
        verbose_name = "Platform"
        verbose_name_plural = "Platformlar"

    def __str__(self):
        return self.name


class Review(models.Model):
    service = models.ForeignKey(FreelanceService, on_delete=models.CASCADE, related_name='reviews', verbose_name="Hizmet")
    platform = models.ForeignKey(Platform, on_delete=models.CASCADE, verbose_name="Çalışılan Platform")
    client_name = models.CharField(max_length=100, verbose_name="Müşteri Adı")

    # Puanlama sistemleri
    communication_rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        verbose_name="İletişim Becerisi (1-5)",
        help_text="Müşteri ile iletişim kalitesi"
    )
    timing_rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        verbose_name="Zamanlama (1-5)",
        help_text="Proje teslim süresi ve zamanında teslim"
    )
    quality_rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        verbose_name="Hizmet Kalitesi (1-5)",
        help_text="Sunulan hizmetin kalitesi"
    )

    comment = models.TextField(verbose_name="Yorum", help_text="Müşterinin deneyimi hakkında yorum")
    project_description = models.TextField(blank=True, null=True, verbose_name="Proje Açıklaması")
    project_image = models.ImageField(
        upload_to='reviews/projects/',
        blank=True,
        null=True,
        verbose_name="Proje Görseli",
        help_text="Hizmet ile ilgili görsel yükleyebilirsiniz"
    )
    is_featured = models.BooleanField(default=False, verbose_name="Öne Çıkan",
                                      help_text="Bu yorumu öne çıkan bölümde göster")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Oluşturulma Tarihi")

    class Meta:
        verbose_name = "Müşteri Yorumu"
        verbose_name_plural = "Müşteri Yorumları"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.client_name} - {self.service.name} ({self.overall_rating}/5)"

    @property
    def overall_rating(self):
        """Genel ortalama puanı hesaplar"""
        try:
            # Eğer değerler None ise 0 olarak kabul et
            communication = self.communication_rating or 0
            timing = self.timing_rating or 0
            quality = self.quality_rating or 0

            if communication == 0 and timing == 0 and quality == 0:
                return 0

            return round((communication + timing + quality) / 3, 1)
        except (AttributeError, TypeError):
            return 0

    @property
    def star_range(self):
        return range(1, 6)

    @property
    def rating_breakdown(self):
        """Puanları kategorilere ayırır"""
        return {
            'İletişim Becerisi': self.communication_rating or 0,
            'Zamanlama': self.timing_rating or 0,
            'Hizmet Kalitesi': self.quality_rating or 0
        }

    def save(self, *args, **kwargs):
        """Model kaydedilmeden önce validasyon"""
        # Puanlama alanları boş bırakılamaz
        if not self.communication_rating:
            self.communication_rating = 1
        if not self.timing_rating:
            self.timing_rating = 1
        if not self.quality_rating:
            self.quality_rating = 1

        super().save(*args, **kwargs)

# Chatbot Modelleri
class ChatSession(models.Model):
    """Chatbot oturum bilgileri"""
    session_id = models.CharField(max_length=100, unique=True)
    user_ip = models.GenericIPAddressField(blank=True, null=True)
    user_agent = models.TextField(blank=True, null=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    message_count = models.IntegerField(default=0)

    class Meta:
        verbose_name = "Chat Oturumu"
        verbose_name_plural = "Chat Oturumları"
        ordering = ['-started_at']

    def __str__(self):
        return f"Session {self.session_id} - {self.started_at.strftime('%Y-%m-%d %H:%M')}"


class ChatMessage(models.Model):
    """Chatbot Messages"""
    MESSAGE_TYPE_CHOICES = [
        ('user', 'User'),
        ('bot', 'Bot'),
        ('system', 'System'),
    ]


    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    # Gemini API ile ilgili alanlar
    model_used = models.CharField(max_length=50, blank=True, null=True)  # gemini-pro vs.
    tokens_used = models.IntegerField(blank=True, null=True)
    response_time = models.FloatField(blank=True, null=True)  # Saniye cinsinden

    class Meta:
        verbose_name = "Chat Mesajı"
        verbose_name_plural = "Chat Mesajları"
        ordering = ['timestamp']
        indexes = [
            models.Index(fields=['session', '-timestamp']),
            models.Index(fields=['message_type', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.get_message_type_display()} - {self.content[:50]}"


class ChatFeedback(models.Model):
    """Chatbot Feedback"""
    RATING_CHOICES = [
        (1, '1 - Very Bad'),
        (2, '2 - Bad'),
        (3, '3 - Average'),
        (4, '4 - Good'),
        (5, '5 - Excellent'),
    ]


    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='feedbacks')
    message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, blank=True, null=True)
    rating = models.IntegerField(choices=RATING_CHOICES, blank=True, null=True)
    feedback_text = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Chat Geri Bildirim"
        verbose_name_plural = "Chat Geri Bildirimleri"
        ordering = ['-created_at']

    def __str__(self):
        return f"Feedback for Session {self.session.session_id}"


class ChatAnalytics(models.Model):
    """Chatbot analitik verileri"""
    date = models.DateField(unique=True)
    total_sessions = models.IntegerField(default=0)
    total_messages = models.IntegerField(default=0)
    avg_messages_per_session = models.FloatField(default=0)
    avg_session_duration = models.FloatField(default=0)  # Dakika cinsinden
    unique_users = models.IntegerField(default=0)

    class Meta:
        verbose_name = "Chat Analitik"
        verbose_name_plural = "Chat Analitikleri"
        ordering = ['-date']

    def __str__(self):
        return f"Analytics for {self.date}"


class GeminiModel(models.Model):
    """Gemini AI modelleri ve API anahtarları için model"""

    MODEL_CHOICES = [
        ('gemini-2.0-flash-exp', 'Gemini 2.0 Flash (Experimental)'),
        ('gemini-2.5-pro', 'Gemini 2.5 Pro'),
        ('gemini-1.5-pro', 'Gemini 1.5 Pro'),
        ('gemini-1.5-flash', 'Gemini 1.5 Flash'),
        ('gemini-pro', 'Gemini Pro'),
    ]

    name = models.CharField(
        max_length=100,
        verbose_name="Model Adı",
        help_text="Gemini model adı"
    )

    model_identifier = models.CharField(
        max_length=50,
        choices=MODEL_CHOICES,
        unique=True,
        verbose_name="Model Tanımlayıcısı",
        help_text="API'de kullanılacak model adı"
    )

    # API anahtarını şifrelenmiş olarak sakla
    encrypted_api_key = models.TextField(
        verbose_name="Şifrelenmiş API Anahtarı",
        help_text="Bu model için Gemini API anahtarı",
        null=True,
        blank=True
    )

    def set_api_key(self, api_key):
        """API anahtarını şifrele ve kaydet"""
        if not api_key:  # Boş string kontrolü ekle
            self.encrypted_api_key = None
            return

        if not hasattr(settings, 'ENCRYPTION_KEY') or not settings.ENCRYPTION_KEY:
            raise ValueError("ENCRYPTION_KEY ayarı bulunamadı veya boş")

        try:
            fernet = Fernet(settings.ENCRYPTION_KEY.encode())
            self.encrypted_api_key = fernet.encrypt(api_key.encode()).decode()
        except Exception as e:
            raise ValueError(f"API anahtarı şifrelenemedi: {str(e)}")

    def get_api_key(self):
        """API anahtarını çöz ve döndür"""
        if not self.encrypted_api_key:
            return None

        if not hasattr(settings, 'ENCRYPTION_KEY') or not settings.ENCRYPTION_KEY:
            raise ValueError("ENCRYPTION_KEY ayarı bulunamadı veya boş")

        try:
            fernet = Fernet(settings.ENCRYPTION_KEY.encode())
            return fernet.decrypt(self.encrypted_api_key.encode()).decode()
        except Exception as e:
            raise ValueError(f"API anahtarı çözülemedi: {str(e)}")

    # Property olarak api_key tanımla
    api_key = property(get_api_key, set_api_key)

    is_active = models.BooleanField(
        default=False,
        verbose_name="Aktif Mi?",
        help_text="Bu model şu anda kullanılıyor mu?"
    )

    is_default = models.BooleanField(
        default=False,
        verbose_name="Varsayılan Mi?",
        help_text="Bu model varsayılan model mi?"
    )

    max_tokens = models.IntegerField(
        default=2048,
        verbose_name="Maksimum Token",
        help_text="Bu model için maksimum token sayısı"
    )

    temperature = models.FloatField(
        default=0.7,
        verbose_name="Sıcaklık",
        help_text="Model yaratıcılık seviyesi (0.0 - 1.0)"
    )

    description = models.TextField(
        blank=True,
        null=True,
        verbose_name="Açıklama",
        help_text="Model hakkında açıklama"
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Oluşturulma Tarihi"
    )

    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name="Güncellenme Tarihi"
    )

    is_enabled = models.BooleanField(
        default=True,
        verbose_name="Etkin Mi?",
        help_text="Model kullanıma açık mı?"
    )

    class Meta:
        verbose_name = "Gemini Model"
        verbose_name_plural = "Gemini Modelleri"
        ordering = ['-is_active', '-is_default', 'name']

    def __str__(self):
        status = " (Aktif)" if self.is_active else ""
        default = " (Varsayılan)" if self.is_default else ""
        return f"{self.name}{status}{default}"

    def clean(self):
        """Model validasyonu"""
        # Sadece bir model aktif olabilir
        if self.is_active:
            other_active = GeminiModel.objects.filter(is_active=True).exclude(pk=self.pk)
            if other_active.exists():
                raise ValidationError("Sadece bir model aktif olabilir!")

        # Sadece bir model varsayılan olabilir
        if self.is_default:
            other_default = GeminiModel.objects.filter(is_default=True).exclude(pk=self.pk)
            if other_default.exists():
                raise ValidationError("Sadece bir model varsayılan olabilir!")

        # Temperature kontrolü
        if not (0.0 <= self.temperature <= 1.0):
            raise ValidationError("Sıcaklık değeri 0.0 ile 1.0 arasında olmalıdır!")

    def save(self, *args, **kwargs):
        self.full_clean()

        # Eğer bu model aktif yapılıyorsa, diğerlerini pasif yap
        if self.is_active:
            GeminiModel.objects.filter(is_active=True).exclude(pk=self.pk).update(is_active=False)

        # Eğer bu model varsayılan yapılıyorsa, diğerlerini varsayılan olmaktan çıkar
        if self.is_default:
            GeminiModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)

        super().save(*args, **kwargs)

    @classmethod
    def get_active_model(cls):
        """Aktif modeli döner"""
        try:
            return cls.objects.get(is_active=True, is_enabled=True)
        except cls.DoesNotExist:
            # Aktif model yoksa varsayılanı dön
            try:
                return cls.objects.get(is_default=True, is_enabled=True)
            except cls.DoesNotExist:
                # Hiçbiri yoksa ilk etkin modeli dön
                return cls.objects.filter(is_enabled=True).first()

    @classmethod
    def get_default_model(cls):
        """Varsayılan modeli döner"""
        try:
            return cls.objects.get(is_default=True, is_enabled=True)
        except cls.DoesNotExist:
            return cls.objects.filter(is_enabled=True).first()

class GeminiModelUsage(models.Model):
    """Model kullanım istatistikleri"""

    model = models.ForeignKey(
        GeminiModel,
        on_delete=models.CASCADE,
        verbose_name="Model"
    )

    session = models.ForeignKey(
        'ChatSession',  # ChatSession modelinizin bulunduğu app adını kullanın
        on_delete=models.CASCADE,
        verbose_name="Oturum",
        null=True,
        blank=True
    )

    tokens_used = models.IntegerField(
        default=0,
        verbose_name="Kullanılan Token"
    )

    response_time = models.FloatField(
        verbose_name="Yanıt Süresi (saniye)"
    )

    success = models.BooleanField(
        default=True,
        verbose_name="Başarılı mı?"
    )

    error_message = models.TextField(
        blank=True,
        null=True,
        verbose_name="Hata Mesajı"
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Kullanım Tarihi"
    )

    class Meta:
        verbose_name = "Model Kullanımı"
        verbose_name_plural = "Model Kullanımları"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.model.name} - {self.created_at.strftime('%d.%m.%Y %H:%M')}"