from django.contrib import admin
from .models import (
    Profile, Education, About, Service, SkillCategory, Skill,
    ProjectCategory, Project, Certification, Experience,
    Achievement, Communication, Resume, ChatSession,
    ChatMessage, ChatFeedback, ChatAnalytics,EmailSettings, ContactMessage
)
from django.utils.safestring import mark_safe
from django import forms
from .models import FreelanceService, Review, Platform
from django.urls import path
from django.shortcuts import render, redirect
from django.http import FileResponse, HttpResponse
from django.conf import settings
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
import os
import shutil
from datetime import datetime

# Profile Admin
@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('name', 'title', 'email', 'phone', 'updated_at')
    list_editable = ('title', 'email', 'phone')
    search_fields = ('name', 'email')
    readonly_fields = ('created_at', 'updated_at')


# Education Admin
@admin.register(Education)
class EducationAdmin(admin.ModelAdmin):
    list_display = ('school', 'title', 'start_date', 'end_date', 'is_current', 'order')
    list_editable = ('is_current', 'order')
    list_filter = ('is_current', 'start_date')
    search_fields = ('school', 'title')
    ordering = ['order', '-start_date']


# About Admin
@admin.register(About)
class AboutAdmin(admin.ModelAdmin):
    list_display = ('__str__', 'years_of_experience', 'updated_at')
    readonly_fields = ('created_at', 'updated_at')
    fieldsets = (
        (None, {
            'fields': ('short_description', 'about', 'years_of_experience')
        }),
        ('Zaman Bilgileri', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


# Service Admin
@admin.register(Service)
class ServiceAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_active', 'order')
    list_editable = ('is_active', 'order')
    list_filter = ('is_active',)
    search_fields = ('name', 'description')
    ordering = ['order']


# Skill Category Admin
@admin.register(SkillCategory)
class SkillCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'order')
    list_editable = ('order',)
    ordering = ['order']


# Skill Admin
@admin.register(Skill)
class SkillAdmin(admin.ModelAdmin):
    list_display = ('name', 'category', 'level', 'percentage', 'order')
    list_editable = ('level', 'percentage', 'order')
    list_filter = ('category', 'level')
    search_fields = ('name', 'category__name')
    ordering = ['category', 'order']


# Project Category Admin
@admin.register(ProjectCategory)
class ProjectCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'order')
    list_editable = ('order',)
    ordering = ['order']


# Project Admin
@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'status', 'is_featured', 'start_date', 'order')
    list_editable = ('status', 'is_featured', 'order')
    list_filter = ('category', 'status', 'is_featured', 'start_date')
    search_fields = ('title', 'description')
    filter_horizontal = ('technologies',)
    ordering = ['-start_date', 'order']
    fieldsets = (
        (None, {
            'fields': ('title', 'category', 'short_description', 'description')
        }),
        ('Medya ve Linkler', {
            'fields': ('image', 'link', 'github_link')
        }),
        ('Proje Detayları', {
            'fields': ('technologies', 'status', 'start_date', 'end_date')
        }),
        ('Görünüm Ayarları', {
            'fields': ('is_featured', 'order'),
            'classes': ('collapse',)
        }),
    )


# Certification Admin
@admin.register(Certification)
class CertificationAdmin(admin.ModelAdmin):
    list_display = ('title', 'institution', 'date', 'expiry_date', 'is_active')
    list_editable = ('is_active',)
    list_filter = ('institution', 'is_active', 'date')
    search_fields = ('title', 'institution', 'credential_id')
    ordering = ['-date']


# Experience Admin
@admin.register(Experience)
class ExperienceAdmin(admin.ModelAdmin):
    list_display = ('job_title', 'company', 'employment_type', 'start_date', 'end_date', 'is_current')
    list_editable = ('employment_type', 'is_current')
    list_filter = ('employment_type', 'is_current', 'start_date')
    search_fields = ('job_title', 'company', 'description')
    filter_horizontal = ('technologies',)
    ordering = ['-start_date']
    fieldsets = (
        ('Temel Bilgiler', {
            'fields': ('job_title', 'company', 'company_logo', 'location')
        }),
        ('Çalışma Detayları', {
            'fields': ('employment_type', 'start_date', 'end_date', 'is_current')
        }),
        ('Açıklamalar', {
            'fields': ('description', 'achievements')
        }),
        ('Teknolojiler', {
            'fields': ('technologies',),
            'classes': ('collapse',)
        }),
    )


# Achievement Admin
@admin.register(Achievement)
class AchievementAdmin(admin.ModelAdmin):
    list_display = ('title', 'organization', 'date', 'is_featured', 'has_certificate', 'certificate_preview')
    list_editable = ('is_featured',)
    list_filter = ('is_featured', 'date', 'organization', 'certificate')
    search_fields = ('title', 'description', 'organization')
    ordering = ['-date']

    # Fieldsets for better organization in admin form
    fieldsets = (
        ('Temel Bilgiler', {
            'fields': ('title', 'description', 'logo')
        }),
        ('Detaylar', {
            'fields': ('date', 'organization', 'link', 'is_featured')
        }),
        ('Sertifika/Belge', {
            'fields': ('certificate',),
            'classes': ('collapse',),  # Başlangıçta kapalı
            'description': 'Başarıya ait sertifika veya belge yükleyebilirsiniz (PDF, JPG, PNG, DOCX vb.)'
        }),
    )

    # Readonly fields for better info display
    readonly_fields = ('certificate_info',)

    def has_certificate(self, obj):
        """Sertifika varlığını gösterir"""
        if obj.certificate:
            return format_html(
                '<span style="color: green;">✓ Var</span>'
            )
        return format_html(
            '<span style="color: red;">✗ Yok</span>'
        )

    has_certificate.boolean = False  # HTML kullanıyoruz
    has_certificate.short_description = 'Sertifika'
    has_certificate.admin_order_field = 'certificate'  # Sırala

    def certificate_preview(self, obj):
        """Sertifika önizlemesi gösterir"""
        if not obj.certificate:
            return "-"

        # Dosya türüne göre önizleme
        if obj.is_image_certificate:
            return format_html(
                '<img src="{}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 5px;" title="{}">',
                obj.certificate.url,
                obj.certificate.name
            )
        elif obj.is_pdf_certificate:
            return format_html(
                '<i class="fas fa-file-pdf" style="font-size: 24px; color: #dc3545;" title="PDF: {}"></i>',
                obj.certificate.name
            )
        else:
            return format_html(
                '<i class="{}" style="font-size: 24px; color: #6c757d;" title="{}"></i>',
                obj.certificate_icon,
                obj.certificate.name
            )

    certificate_preview.short_description = 'Önizleme'

    def certificate_info(self, obj):
        """Sertifika detaylı bilgileri gösterir"""
        if not obj.certificate:
            return "Sertifika yüklenmemiş"

        # Dosya boyutunu hesapla
        file_size = obj.certificate.size
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"

        return format_html(
            """
            <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">
                <p><strong>Dosya Adı:</strong> {}</p>
                <p><strong>Dosya Türü:</strong> {}</p>
                <p><strong>Dosya Boyutu:</strong> {}</p>
                <p><strong>Yükleme Tarihi:</strong> Mevcut değil</p>
                <p><a href="{}" target="_blank" style="color: #007bff;">Dosyayı Görüntüle/İndir</a></p>
            </div>
            """,
            obj.certificate.name,
            obj.certificate_extension.upper() if obj.certificate_extension else "Bilinmiyor",
            size_str,
            obj.certificate.url
        )

    certificate_info.short_description = 'Sertifika Bilgileri'

    # Custom actions
    actions = ['mark_as_featured', 'mark_as_not_featured', 'export_achievements']

    def mark_as_featured(self, request, queryset):
        """Seçili başarıları öne çıkarılmış olarak işaretle"""
        updated = queryset.update(is_featured=True)
        self.message_user(
            request,
            f'{updated} başarı öne çıkarılmış olarak işaretlendi.'
        )

    mark_as_featured.short_description = "Seçili başarıları öne çıkar"

    def mark_as_not_featured(self, request, queryset):
        """Seçili başarıları öne çıkarılmış olmaktan çıkar"""
        updated = queryset.update(is_featured=False)
        self.message_user(
            request,
            f'{updated} başarı öne çıkarılmış olmaktan çıkarıldı.'
        )

    mark_as_not_featured.short_description = "Seçili başarıları öne çıkarmaktan çıkar"

    def export_achievements(self, request, queryset):
        """Seçili başarıları CSV olarak dışa aktar"""
        import csv
        from django.http import HttpResponse

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="basarilar.csv"'

        writer = csv.writer(response)
        writer.writerow(['Başlık', 'Organizasyon', 'Tarih', 'Öne Çıkan', 'Sertifika Var'])

        for achievement in queryset:
            writer.writerow([
                achievement.title,
                achievement.organization or '',
                achievement.date or '',
                'Evet' if achievement.is_featured else 'Hayır',
                'Evet' if achievement.certificate else 'Hayır'
            ])

        return response

    export_achievements.short_description = "Seçili başarıları CSV olarak dışa aktar"

    # Admin form'da JavaScript eklemek için (isteğe bağlı)
    class Media:
        css = {
            'all': ('admin/css/achievements_admin.css',)
        }
        js = ('admin/js/achievements_admin.js',)


# Communication Admin
@admin.register(Communication)
class CommunicationAdmin(admin.ModelAdmin):
    list_display = ('title', 'type', 'value', 'is_public', 'order')
    list_editable = ('is_public', 'order')
    list_filter = ('type', 'is_public')
    search_fields = ('title', 'value')
    ordering = ['order']


# Resume Admin
@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ('title', 'language', 'is_active', 'uploaded_at')
    list_editable = ('is_active',)
    list_filter = ('language', 'is_active', 'uploaded_at')
    ordering = ['-uploaded_at']


from django.utils.html import format_html

@admin.register(Platform)
class PlatformAdmin(admin.ModelAdmin):
    list_display = ['name', 'logo_preview', 'color', 'link_preview']
    list_filter = ['name']
    search_fields = ['name']

    def logo_preview(self, obj):
        if obj.platform_logo:
            return format_html('<img src="{}" width="30" height="30" style="border-radius: 4px;" />',
                               obj.platform_logo.url)
        return "Logo Yok"
    logo_preview.short_description = 'Logo Önizleme'

    def link_preview(self, obj):
        if obj.link:
            return format_html('<a href="{}" target="_blank">{}</a>', obj.link, obj.link)
        return "Link Yok"
    link_preview.short_description = 'Platform Linki'


@admin.register(FreelanceService)
class FreelanceServiceAdmin(admin.ModelAdmin):
    list_display = ['name', 'image_preview', 'service_link', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'description']

    def image_preview(self, obj):
        if obj.service_image:
            return format_html('<img src="{}" width="40" height="30" style="border-radius: 4px; object-fit: cover;" />',
                               obj.service_image.url)
        return "Görsel Yok"

    image_preview.short_description = 'Görsel Önizleme'


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ['client_name', 'service', 'platform', 'get_overall_rating', 'is_featured', 'created_at']
    list_filter = ['service', 'platform', 'is_featured', 'created_at', 'communication_rating', 'timing_rating',
                   'quality_rating']
    search_fields = ['client_name', 'comment']
    list_editable = ['is_featured']
    readonly_fields = ['created_at']

    def get_overall_rating(self, obj):
        """Admin listesinde genel puanı göster"""
        return f"{obj.overall_rating}/5"

    get_overall_rating.short_description = 'Genel Puan'
    get_overall_rating.admin_order_field = 'communication_rating'

    fieldsets = (
        ('Temel Bilgiler', {
            'fields': ('service', 'platform', 'client_name', 'is_featured'),
            'description': 'Müşteri ve hizmet bilgileri'
        }),
        ('Puanlama Sistemi', {
            'fields': ('communication_rating', 'timing_rating', 'quality_rating'),
            'description': 'Her kategori için 1-5 arası puan verin (1: Çok Kötü, 2: Kötü, 3: Orta, 4: İyi, 5: Mükemmel)'
        }),
        ('İçerik ve Medya', {
            'fields': ('comment', 'project_description', 'project_image'),
            'description': 'Yorum metni ve proje görselleri'
        }),
        ('Sistem Bilgileri', {
            'fields': ('created_at',),
            'classes': ('collapse',),
            'description': 'Otomatik oluşturulan sistem bilgileri'
        })
    )

    def get_changeform_initial_data(self, request):
        return {
            'communication_rating': 5,
            'timing_rating': 5,
            'quality_rating': 5,
            'is_featured': False
        }

class EmailSettingsAdminForm(forms.ModelForm):
    """EmailSettings için özel admin form"""
    raw_password = forms.CharField(
        label='Uygulama Şifresi',
        widget=forms.PasswordInput(attrs={
            'placeholder': 'Yeni şifre girmek için buraya yazın'
        }),
        required=False,
        help_text='Gmail uygulama şifrenizi buraya girin. Boş bırakırsanız mevcut şifre değişmez.'
    )

    class Meta:
        model = EmailSettings
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # encrypted_password alanını gizle
        if 'encrypted_password' in self.fields:
            self.fields['encrypted_password'].widget = forms.HiddenInput()

    def save(self, commit=True):
        instance = super().save(commit=False)
        raw_password = self.cleaned_data.get('raw_password')
        if raw_password:
            instance.set_password(raw_password)
        if commit:
            instance.save()
        return instance


@admin.register(EmailSettings)
class EmailSettingsAdmin(admin.ModelAdmin):
    form = EmailSettingsAdminForm
    list_display = ['name', 'email', 'smtp_server', 'smtp_port', 'is_active', 'has_password_display', 'created_at']
    list_filter = ['is_active', 'use_tls', 'created_at']
    search_fields = ['name', 'email']
    readonly_fields = ['created_at', 'updated_at', 'password_status']

    fieldsets = (
        ('Genel Bilgiler', {
            'fields': ('name', 'email', 'is_active')
        }),
        ('SMTP Ayarları', {
            'fields': ('smtp_server', 'smtp_port', 'use_tls')
        }),
        ('Şifre Yönetimi', {
            'fields': ('password_status', 'raw_password')
        }),
        ('Zaman Bilgileri', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def has_password_display(self, obj):
        """Şifre durumunu gösterir"""
        if obj.has_password:
            return format_html('<span style="color: green;">✓ Şifre Tanımlı</span>')
        return format_html('<span style="color: red;">✗ Şifre Yok</span>')

    has_password_display.short_description = 'Şifre Durumu'

    def password_status(self, obj):
        """Şifre durumu read-only alanı"""
        if obj and obj.has_password:
            return "✓ Şifre tanımlı ve şifreli olarak saklanıyor"
        return "⚠ Şifre tanımlanmamış"

    password_status.short_description = 'Mevcut Şifre Durumu'

@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'subject', 'status', 'is_spam', 'created_at', 'message_preview']
    list_filter = ['status', 'is_spam', 'created_at']
    search_fields = ['name', 'email', 'subject', 'message']
    readonly_fields = ['created_at', 'updated_at', 'ip_address', 'user_agent']
    list_per_page = 20
    date_hierarchy = 'created_at'

    fieldsets = (
        ('Gönderen Bilgileri', {
            'fields': ('name', 'email')
        }),
        ('Mesaj Detayları', {
            'fields': ('subject', 'message', 'status', 'is_spam')
        }),
        ('Teknik Bilgiler', {
            'fields': ('ip_address', 'user_agent'),
            'classes': ('collapse',)
        }),
        ('Zaman Bilgileri', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def message_preview(self, obj):
        """Mesajın kısa önizlemesini gösterir"""
        preview = obj.message[:100]
        if len(obj.message) > 100:
            preview += "..."
        return preview

    message_preview.short_description = 'Mesaj Önizleme'

    def get_queryset(self, request):
        """Queryset'i optimize et"""
        return super().get_queryset(request).select_related()

    actions = ['mark_as_read', 'mark_as_replied', 'mark_as_spam', 'mark_as_archived']

    def mark_as_read(self, request, queryset):
        """Seçili mesajları okundu olarak işaretle"""
        updated = queryset.update(status='read')
        self.message_user(request, f'{updated} mesaj okundu olarak işaretlendi.')

    mark_as_read.short_description = 'Seçili mesajları okundu olarak işaretle'

    def mark_as_replied(self, request, queryset):
        """Seçili mesajları yanıtlandı olarak işaretle"""
        updated = queryset.update(status='replied')
        self.message_user(request, f'{updated} mesaj yanıtlandı olarak işaretlendi.')

    mark_as_replied.short_description = 'Seçili mesajları yanıtlandı olarak işaretle'

    def mark_as_spam(self, request, queryset):
        """Seçili mesajları spam olarak işaretle"""
        updated = queryset.update(is_spam=True, status='archived')
        self.message_user(request, f'{updated} mesaj spam olarak işaretlendi.')

    mark_as_spam.short_description = 'Seçili mesajları spam olarak işaretle'

    def mark_as_archived(self, request, queryset):
        """Seçili mesajları arşivle"""
        updated = queryset.update(status='archived')
        self.message_user(request, f'{updated} mesaj arşivlendi.')

    mark_as_archived.short_description = 'Seçili mesajları arşivle'

class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ['timestamp', 'message_preview', 'model_used', 'response_time']
    fields = ['message_type', 'message_preview', 'model_used', 'response_time', 'timestamp']
    can_delete = False

    def message_preview(self, obj):
        if obj.message_type == 'user':
            icon = "👤"
            style = "background-color: #e3f2fd; padding: 8px; border-radius: 8px; margin: 2px 0;"
        else:
            icon = "🤖"
            style = "background-color: #f3e5f5; padding: 8px; border-radius: 8px; margin: 2px 0;"

        content = obj.content[:100] + "..." if len(obj.content) > 100 else obj.content
        return format_html(
            '<div style="{}">{} <strong>{}:</strong><br>{}</div>',
            style, icon, obj.get_message_type_display(), content
        )

    message_preview.short_description = 'Mesaj İçeriği'

    def has_add_permission(self, request, obj=None):
        return False


class ChatFeedbackInline(admin.TabularInline):
    model = ChatFeedback
    extra = 0
    readonly_fields = ['rating', 'feedback_text', 'created_at']
    can_delete = False

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ['session_preview', 'user_info', 'session_stats', 'duration', 'status_badge']
    list_filter = ['is_active', 'started_at', 'ended_at']
    search_fields = ['session_id', 'user_ip']
    readonly_fields = ['session_id', 'started_at', 'session_summary']
    ordering = ['-started_at']
    inlines = [ChatMessageInline, ChatFeedbackInline]

    fieldsets = (
        ('Session Bilgileri', {
            'fields': ('session_id', 'user_ip', 'started_at', 'ended_at', 'is_active')
        }),
        ('Session Özeti', {
            'fields': ('session_summary',),
            'classes': ('wide',)
        }),
    )

    def session_preview(self, obj):
        return format_html(
            '<strong>{}</strong><br><small style="color: #666;">{}</small>',
            obj.session_id[:12] + '...',
            obj.started_at.strftime('%d.%m.%Y %H:%M')
        )

    session_preview.short_description = 'Session'

    def user_info(self, obj):
        return format_html(
            '🌐 {}<br><small>IP: {}</small>',
            obj.user_ip[:15] + '...' if len(obj.user_ip) > 15 else obj.user_ip,
            obj.user_ip
        )

    user_info.short_description = 'Kullanıcı'

    def session_stats(self, obj):
        message_count = obj.messages.count()
        feedback_count = obj.feedbacks.count()
        return format_html(
            '💬 {} mesaj<br>⭐ {} geri bildirim',
            message_count, feedback_count
        )

    session_stats.short_description = 'İstatistikler'

    def duration(self, obj):
        if obj.ended_at and obj.started_at:
            duration = obj.ended_at - obj.started_at
            minutes = int(duration.total_seconds() / 60)
            return f"⏱️ {minutes} dakika"
        return "⏱️ Devam ediyor"

    duration.short_description = 'Süre'

    def status_badge(self, obj):
        if obj.is_active:
            return format_html(
                '<span style="background-color: #4caf50; color: white; padding: 4px 8px; '
                'border-radius: 12px; font-size: 11px;">🟢 Aktif</span>'
            )
        else:
            return format_html(
                '<span style="background-color: #f44336; color: white; padding: 4px 8px; '
                'border-radius: 12px; font-size: 11px;">🔴 Tamamlandı</span>'
            )

    status_badge.short_description = 'Durum'

    def session_summary(self, obj):
        messages = obj.messages.order_by('timestamp')[:10]

        if not messages:
            return "Henüz mesaj yok."

        html = """
        <div style="
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
            padding: 15px;
            border-radius: 8px;
            background: #ffffff;
        ">
        """

        for msg in messages:
            if msg.message_type == 'user':
                html += format_html("""
                    <div style="
                        margin: 10px 0;
                        padding: 10px 15px;
                        background: #f8f9fa;
                        color: #212529;
                        border-radius: 8px 8px 0 8px;
                        max-width: 80%;
                        margin-left: auto;
                        border: 1px solid #dee2e6;
                    ">
                        <strong style="color: #495057;">👤 Kullanıcı:</strong><br>
                        {}<br>
                        <small style="color: #6c757d;">{}</small>
                    </div>
                    """,
                                    msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                                    msg.timestamp.strftime('%H:%M')
                                    )
            else:
                html += format_html("""
                    <div style="
                        margin: 10px 0;
                        padding: 10px 15px;
                        background: #ffffff;
                        color: #212529;
                        border-radius: 8px 8px 8px 0;
                        max-width: 80%;
                        margin-right: auto;
                        border: 1px solid #dee2e6;
                    ">
                        <strong style="color: #495057;">🤖 Bot ({}):</strong><br>
                        {}<br>
                        <small style="color: #6c757d;">{} • {}ms</small>
                    </div>
                    """,
                                    msg.model_used or 'N/A',
                                    msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                                    msg.timestamp.strftime('%H:%M'),
                                    msg.response_time or 0
                                    )

        if messages.count() > 10:
            html += """
            <div style="
                text-align: center;
                margin-top: 12px;
                font-style: italic;
                color: #6c757d;
            ">
                ... ve {} mesaj daha
            </div>
            """.format(messages.count() - 10)

        html += '</div>'
        return mark_safe(html)

    session_summary.short_description = 'Sohbet Geçmişi'

    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related('messages', 'feedbacks')


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['message_preview', 'session_link', 'message_type_badge', 'timestamp', 'model_info']
    list_filter = ['message_type', 'timestamp', 'model_used']
    search_fields = ['content', 'session__session_id']
    readonly_fields = ['timestamp', 'formatted_content']
    ordering = ['-timestamp']

    fieldsets = (
        ('Mesaj Bilgileri', {
            'fields': ('session', 'message_type', 'model_used', 'response_time', 'timestamp')
        }),
        ('Mesaj İçeriği', {
            'fields': ('formatted_content',),
            'classes': ('wide',)
        }),
    )

    def message_preview(self, obj):
        icon = "👤" if obj.message_type == 'user' else "🤖"
        preview = obj.content[:60] + "..." if len(obj.content) > 60 else obj.content
        return format_html('{} {}', icon, preview)

    message_preview.short_description = 'Mesaj Önizleme'

    def session_link(self, obj):
        url = reverse('admin:resume_chatsession_change',
                      args=[obj.session.pk])  # your_app kısmını kendi app isminizle değiştirin
        return format_html('<a href="{}">📋 {}</a>', url, obj.session.session_id[:12])

    session_link.short_description = 'Session'

    def message_type_badge(self, obj):
        if obj.message_type == 'user':
            return format_html(
                '<span style="background-color: #2196f3; color: white; padding: 2px 8px; '
                'border-radius: 12px; font-size: 11px;">👤 Kullanıcı</span>'
            )
        else:
            return format_html(
                '<span style="background-color: #9c27b0; color: white; padding: 2px 8px; '
                'border-radius: 12px; font-size: 11px;">🤖 Bot</span>'
            )

    message_type_badge.short_description = 'Tür'

    def model_info(self, obj):
        if obj.model_used:
            return format_html('🧠 {} ({}ms)', obj.model_used, obj.response_time or 0)
        return '-'

    model_info.short_description = 'Model'

    def formatted_content(self, obj):
        style = """
            padding: 12px 15px;
            border-radius: 6px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 8px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        """

        if obj.message_type == 'user':
            style += """
                background-color: #f8f9fa;
                border-left: 4px solid #6c757d;
                color: #212529;
            """
        else:
            style += """
                background-color: #e9ecef;
                border-left: 4px solid #495057;
                color: #212529;
            """

        return format_html(
            '<div style="{}">{}</div>',
            style,
            obj.content.replace('\n', '<br>')
        )
    formatted_content.short_description = 'Mesaj İçeriği'

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('session')


@admin.register(ChatFeedback)
class ChatFeedbackAdmin(admin.ModelAdmin):
    list_display = ['session_link', 'rating_stars', 'feedback_preview', 'created_at']
    list_filter = ['rating', 'created_at']
    search_fields = ['feedback_text', 'session__session_id']
    readonly_fields = ['created_at']
    ordering = ['-created_at']

    def session_link(self, obj):
        url = reverse('admin:resume_chatsession_change', args=[obj.session.pk])  # your_app kısmını değiştirin
        return format_html('<a href="{}">📋 {}</a>', url, obj.session.session_id[:12])

    session_link.short_description = 'Session'

    def rating_stars(self, obj):
        stars = '⭐' * obj.rating + '☆' * (5 - obj.rating)
        return format_html('<span style="font-size: 16px;">{}</span> ({})', stars, obj.rating)

    rating_stars.short_description = 'Değerlendirme'

    def feedback_preview(self, obj):
        if obj.feedback_text:
            preview = obj.feedback_text[:80] + "..." if len(obj.feedback_text) > 80 else obj.feedback_text
            return format_html(
                '<div style="padding: 5px; background-color: #fff3e0; border-radius: 4px; '
                'border-left: 3px solid #ff9800;">{}</div>', preview
            )
        return format_html('<em style="color: #999;">Yorum yok</em>')

    feedback_preview.short_description = 'Geri Bildirim'


@admin.register(ChatAnalytics)
class ChatAnalyticsAdmin(admin.ModelAdmin):
    list_display = ['date', 'daily_stats', 'session_metrics', 'user_metrics']
    list_filter = ['date']
    ordering = ['-date']

    def daily_stats(self, obj):
        return format_html(
            '📊 <strong>{}</strong> session<br>💬 <strong>{}</strong> mesaj',
            obj.total_sessions, obj.total_messages
        )

    daily_stats.short_description = 'Günlük İstatistikler'

    def session_metrics(self, obj):
        return format_html(
            '📈 Ort. <strong>{:.1f}</strong> mesaj/session<br>⏱️ Ort. <strong>{:.1f}</strong> dk süre',
            obj.avg_messages_per_session, obj.avg_session_duration
        )

    session_metrics.short_description = 'Session Metrikleri'

    def user_metrics(self, obj):
        return format_html(
            '👥 <strong>{}</strong> benzersiz kullanıcı',
            obj.unique_users
        )

    user_metrics.short_description = 'Kullanıcı Metrikleri'

    # Sadece okuma izni
    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


# admin.py dosyanıza eklenecek admin konfigürasyonları

from django.contrib import admin
from django.utils.html import format_html
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.urls import path, reverse
from django.shortcuts import render
from .models import GeminiModel, GeminiModelUsage


class GeminiModelForm(forms.ModelForm):
    """Özel form - API anahtarı girişi için"""
    api_key_input = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.PasswordInput(attrs={'placeholder': 'API anahtarını değiştirmek için girin'}),
        label="API Anahtarı",
        help_text="Yeni API anahtarı girmek için bu alanı kullanın (mevcut anahtar şifrelenmiş olarak saklanır)"
    )

    class Meta:
        model = GeminiModel
        exclude = ['encrypted_api_key']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mevcut instance varsa ve API anahtarı varsa placeholder göster
        if self.instance and self.instance.pk and self.instance.encrypted_api_key:
            self.fields['api_key_input'].widget.attrs['placeholder'] = '••••••••••••••••'

    def save(self, commit=True):
        instance = super().save(commit=False)

        # Eğer yeni API anahtarı girilmişse, onu kaydet
        api_key_input = self.cleaned_data.get('api_key_input')
        if api_key_input:
            instance.set_api_key(api_key_input)

        if commit:
            instance.save()
        return instance


@admin.register(GeminiModel)
class GeminiModelAdmin(admin.ModelAdmin):
    form = GeminiModelForm

    list_display = [
        'name',
        'model_identifier',
        'status_display',
        'temperature',
        'max_tokens',
        'api_key_status',
        'created_at',
        'action_buttons'
    ]

    list_filter = [
        'is_active',
        'is_default',
        'is_enabled',
        'model_identifier',
        'created_at'
    ]

    search_fields = ['name', 'model_identifier', 'description']

    readonly_fields = ['created_at', 'updated_at', 'api_key_display']

    fieldsets = (
        ('Model Bilgileri', {
            'fields': ('name', 'model_identifier', 'description')
        }),
        ('API Konfigürasyonu', {
            'fields': ('api_key_input', 'api_key_display', 'max_tokens', 'temperature'),
            'classes': ('collapse',)
        }),
        ('Durum Ayarları', {
            'fields': ('is_enabled', 'is_active', 'is_default')
        }),
        ('Tarih Bilgileri', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    actions = ['activate_model', 'deactivate_model', 'set_as_default']

    def api_key_display(self, obj):
        """API anahtarı durumunu göster"""
        if obj and obj.encrypted_api_key:
            return format_html('<span style="color: green;">✓ API Anahtarı Mevcut (Şifrelenmiş)</span>')
        return format_html('<span style="color: red;">✗ API Anahtarı Yok</span>')

    api_key_display.short_description = "API Anahtarı Durumu"

    def api_key_status(self, obj):
        """Liste görünümü için API anahtarı durumu"""
        if obj and obj.encrypted_api_key:
            return format_html('<span style="color: green;">✓</span>')
        return format_html('<span style="color: red;">✗</span>')

    api_key_status.short_description = "API"

    def status_display(self, obj):
        """Durum göstergesi"""
        status_html = ""
        if obj.is_active:
            status_html += '<span style="color: green; font-weight: bold;">●</span> AKTİF '
        if obj.is_default:
            status_html += '<span style="color: blue; font-weight: bold;">★</span> VARSAYILAN '
        if not obj.is_enabled:
            status_html += '<span style="color: red; font-weight: bold;">✗</span> PASİF'
        if not status_html:
            status_html = '<span style="color: gray;">○</span> Beklemede'
        return format_html(status_html)

    status_display.short_description = "Durum"

    def action_buttons(self, obj):
        """Eylem butonları"""
        buttons = []

        if not obj.is_active and obj.is_enabled:
            activate_url = reverse('admin:activate_gemini_model', args=[obj.pk])
            buttons.append(
                f'<a class="button" href="{activate_url}" style="background-color: green; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px; margin: 2px;">Aktif Et</a>')

        if obj.is_active:
            deactivate_url = reverse('admin:deactivate_gemini_model', args=[obj.pk])
            buttons.append(
                f'<a class="button" href="{deactivate_url}" style="background-color: orange; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px; margin: 2px;">Pasif Et</a>')

        if not obj.is_default and obj.is_enabled:
            default_url = reverse('admin:set_default_gemini_model', args=[obj.pk])
            buttons.append(
                f'<a class="button" href="{default_url}" style="background-color: blue; color: white; padding: 5px 10px; text-decoration: none; border-radius: 3px; margin: 2px;">Varsayılan Yap</a>')

        return format_html(' '.join(buttons))

    action_buttons.short_description = "İşlemler"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('activate/<int:model_id>/', self.activate_model_view, name='activate_gemini_model'),
            path('deactivate/<int:model_id>/', self.deactivate_model_view, name='deactivate_gemini_model'),
            path('set-default/<int:model_id>/', self.set_default_model_view, name='set_default_gemini_model'),
        ]
        return custom_urls + urls

    def activate_model_view(self, request, model_id):
        """Model aktivasyon görünümü"""
        try:
            model = GeminiModel.objects.get(id=model_id)
            # Diğer aktif modelleri pasif et
            GeminiModel.objects.filter(is_active=True).update(is_active=False)
            # Bu modeli aktif et
            model.is_active = True
            model.save()
            messages.success(request, f"{model.name} modeli başarıyla aktif edildi.")
        except GeminiModel.DoesNotExist:
            messages.error(request, "Model bulunamadı.")

        return HttpResponseRedirect(reverse('admin:resume_geminimodel_changelist'))

    def deactivate_model_view(self, request, model_id):
        """Model deaktivasyon görünümü"""
        try:
            model = GeminiModel.objects.get(id=model_id)
            model.is_active = False
            model.save()
            messages.success(request, f"{model.name} modeli başarıyla pasif edildi.")
        except GeminiModel.DoesNotExist:
            messages.error(request, "Model bulunamadı.")

        return HttpResponseRedirect(reverse('admin:resume_geminimodel_changelist'))

    def set_default_model_view(self, request, model_id):
        """Varsayılan model ayarlama görünümü"""
        try:
            model = GeminiModel.objects.get(id=model_id)
            # Diğer varsayılan modelleri kaldır
            GeminiModel.objects.filter(is_default=True).update(is_default=False)
            # Bu modeli varsayılan yap
            model.is_default = True
            model.save()
            messages.success(request, f"{model.name} modeli varsayılan olarak ayarlandı.")
        except GeminiModel.DoesNotExist:
            messages.error(request, "Model bulunamadı.")

        return HttpResponseRedirect(reverse('admin:resume_geminimodel_changelist'))

    def activate_model(self, request, queryset):
        """Toplu model aktivasyonu"""
        if queryset.count() > 1:
            self.message_user(request, "Sadece bir model aktif olabilir!", level=messages.ERROR)
            return

        model = queryset.first()
        # Diğer aktif modelleri pasif et
        GeminiModel.objects.filter(is_active=True).exclude(pk=model.pk).update(is_active=False)
        # Bu modeli aktif et
        model.is_active = True
        model.save()
        self.message_user(request, f"{model.name} modeli aktif edildi.")

    activate_model.short_description = "Seçili modeli aktif et"

    def deactivate_model(self, request, queryset):
        """Toplu model deaktivasyonu"""
        updated = queryset.update(is_active=False)
        self.message_user(request, f"{updated} model pasif edildi.")

    deactivate_model.short_description = "Seçili modelleri pasif et"

    def set_as_default(self, request, queryset):
        """Varsayılan model olarak ayarla"""
        if queryset.count() > 1:
            self.message_user(request, "Sadece bir model varsayılan olabilir!", level=messages.ERROR)
            return

        model = queryset.first()
        # Diğer varsayılan modelleri kaldır
        GeminiModel.objects.filter(is_default=True).exclude(pk=model.pk).update(is_default=False)
        # Bu modeli varsayılan yap
        model.is_default = True
        model.save()
        self.message_user(request, f"{model.name} modeli varsayılan olarak ayarlandı.")

    set_as_default.short_description = "Seçili modeli varsayılan yap"


@admin.register(GeminiModelUsage)
class GeminiModelUsageAdmin(admin.ModelAdmin):
    list_display = [
        'model',
        'session',
        'tokens_used',
        'response_time',
        'success_display',
        'created_at'
    ]

    list_filter = [
        'success',
        'model',
        'created_at'
    ]

    search_fields = ['model__name', 'error_message']

    readonly_fields = ['created_at']

    date_hierarchy = 'created_at'

    def success_display(self, obj):
        """Başarı durumu göstergesi"""
        if obj.success:
            return format_html('<span style="color: green; font-weight: bold;">✓ Başarılı</span>')
        else:
            return format_html('<span style="color: red; font-weight: bold;">✗ Hatalı</span>')

    success_display.short_description = "Durum"

    def has_add_permission(self, request):
        """Kullanım kayıtları manuel olarak eklenemez"""
        return False

    def has_change_permission(self, request, obj=None):
        """Kullanım kayıtları düzenlenemez"""
        return False


# Admin Site Customization
admin.site.site_header = "🤖 Chatbot Yönetim Paneli"
admin.site.site_title = "Chatbot Admin"
admin.site.index_title = "Chatbot Yönetim Paneline Hoş Geldiniz"

