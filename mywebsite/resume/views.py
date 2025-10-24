from django.views.generic import TemplateView
from django.http import JsonResponse, HttpResponse, Http404
from django.contrib import messages
from django.core.paginator import Paginator
from django.conf import settings
import os
from .models import ContactMessage, EmailSettings
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from .models import (
    Profile, Education, About, Service, Skill, Project,
    Certification, Experience, Achievement, Communication, Resume,ProjectCategory,SkillCategory
)
from django.views.decorators.http import require_http_methods
import uuid
from django.utils import timezone
from django.shortcuts import redirect
import json
import logging
from django.views.decorators.csrf import csrf_exempt
from .models import ChatSession, ChatMessage, ChatFeedback, GeminiModel
from .utils import get_gemini_service, get_user_context, create_system_prompt
from django.db.models import Q
from django.core.cache import cache
from django.db.models import Avg, Count
from .models import FreelanceService, Review, Platform
from dateutil.relativedelta import relativedelta
from django.db.models import Min
from datetime import datetime

logger = logging.getLogger(__name__)

class HomeView(TemplateView):
    template_name = 'resume/home.html'

    def calculate_total_experience(self):
        """Toplam deneyim süresini hesaplar"""
        experiences = Experience.objects.all()

        if not experiences.exists():
            return 0

        # En erken başlangıç tarihini bul
        first_start_date = experiences.aggregate(Min('start_date'))['start_date__min']

        if not first_start_date:
            return 0

        # Bugünün tarihi
        today = datetime.now().date()

        # Toplam süreyi hesapla
        total_diff = relativedelta(today, first_start_date)
        total_years = total_diff.years

        # Eğer 6 aydan fazla ise bir sonraki yıla yuvarla
        if total_diff.months >= 6:
            total_years += 1

        return max(total_years, 1)  # En az 1 yıl göster

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({
            'profile': Profile.objects.first(),
            'about': About.objects.first(),
            'FreelanceService': FreelanceService.objects.all()[:3],  # İlk 3 freelance servisini göster
            'services':Service.objects.all()[:3],
            'skills': Skill.objects.all()[:6],  # İlk 6 beceriyi göster
            'recent_projects': Project.objects.all()[:3],  # Son 3 projeyi göster
            'communications': Communication.objects.all(),
            'available_resumes': Resume.objects.filter(is_active=True),  # Aktif özgeçmişleri ekle
            'total_experience_years': self.calculate_total_experience(),  # Dinamik deneyim süresi
        })
        return context

class AboutView(TemplateView):
    template_name = 'resume/about.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({
            'about': About.objects.first(),
            'profile': Profile.objects.first(),
            'skills': Skill.objects.all(),
            'services': Service.objects.all(),
        })
        return context

from datetime import date

class EducationView(TemplateView):
    template_name = 'resume/education.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        educations = Education.objects.all().order_by('-id')
        if educations.exists():
            first_edu = educations.last()
            context['year_diff'] = date.today().year - first_edu.start_date.year
        else:
            context['year_diff'] = 0
        context['educations'] = educations
        return context

class ExperienceView(TemplateView):
    template_name = 'resume/experience.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['experiences'] = Experience.objects.all().order_by('-start_date')
        return context


class SkillsView(TemplateView):
    template_name = 'resume/skills.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        skills = Skill.objects.all().order_by("category__order", "order", "name")



        # İleri + Uzman seviye yetenekler (daha anlamlı)
        advanced_skills_count = skills.filter(
            level__in=['advanced', 'expert']
        ).count()

        # Kategori sayısı
        categories_count = SkillCategory.objects.filter(
            skills__isnull=False
        ).distinct().count()

        # Ortalama yüzde (sadece yüzdesi olan yetenekler)
        avg_percentage = skills.filter(
            percentage__isnull=False
        ).aggregate(avg=Avg('percentage'))['avg']

        if avg_percentage:
            avg_percentage = round(avg_percentage)
        else:
            avg_percentage = 0

        # İkon bulunan yetenekler (görsellik için)
        skills_with_icons = skills.filter(
            icon__isnull=False
        ).exclude(icon='').count()

        context.update({
            'skills': skills,
            'advanced_skills_count': advanced_skills_count,
            'categories_count': categories_count,
            'avg_percentage': avg_percentage,
            'skills_with_icons': skills_with_icons,
        })

        return context


class ProjectsView(TemplateView):
    template_name = 'resume/projects.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Tüm projeler (sayfalama için)
        projects = Project.objects.all().order_by('order', '-start_date')
        paginator = Paginator(projects, 6)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        context['page_obj'] = page_obj

        # Öne çıkan projeler (is_featured=True olanlar)
        featured_projects = Project.objects.filter(
            is_featured=True
        ).order_by('order', '-start_date')[:6]  # İlk 3 öne çıkan proje
        context['featured_projects'] = featured_projects

        # Kategoriler
        categories = ProjectCategory.objects.all().order_by('order')
        context['categories'] = categories

        return context

class CertificationsView(TemplateView):
    template_name = 'resume/certifications.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Artık order + date ile sıralı
        certifications = Certification.objects.all().order_by('order', '-date')
        context['certifications'] = certifications

        # Aktif sertifika sayısı hesapla
        active_certifications = certifications.filter(
            Q(is_active=True) &
            (Q(expiry_date__isnull=True) | Q(expiry_date__gte=date.today()))
        )
        context['active_certifications_count'] = active_certifications.count()

        # Sertifikasyon geçmişi süresi
        if certifications.exists():
            first_cert_date = certifications.aggregate(Min('date'))['date__min']
            if first_cert_date:
                years_diff = date.today().year - first_cert_date.year
                if (date.today().month, date.today().day) < (first_cert_date.month, first_cert_date.day):
                    years_diff -= 1
                context['certification_years'] = max(years_diff, 1)
            else:
                context['certification_years'] = 0
        else:
            context['certification_years'] = 0

        # Farklı kurum sayısı
        unique_institutions = certifications.values_list('institution', flat=True).distinct().count()
        context['unique_institutions'] = unique_institutions

        return context


class AchievementsView(TemplateView):
    template_name = 'resume/achievements.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        achievements = Achievement.objects.all()

        # Başarı süresi hesaplama (ilk başarıdan itibaren)
        if achievements.exists():
            # En eski başarının tarihini bul
            first_achievement_date = achievements.exclude(date__isnull=True).aggregate(
                first_date=Min('date')
            )['first_date']

            if first_achievement_date:
                context['achievement_years'] = date.today().year - first_achievement_date.year
                # Eğer bu yılın ayı/günü henüz geçmemişse 1 eksilt
                if (date.today().month, date.today().day) < (first_achievement_date.month, first_achievement_date.day):
                    context['achievement_years'] -= 1
                context['achievement_years'] = max(context['achievement_years'], 1)  # En az 1 yıl
            else:
                context['achievement_years'] = 0
        else:
            context['achievement_years'] = 0

        # Öne çıkan başarıları say
        featured_count = achievements.filter(is_featured=True).count()

        # Farklı organizasyonları say (null değerleri hariç tut)
        unique_orgs = achievements.exclude(
            organization__isnull=True
        ).exclude(
            organization__exact=''
        ).values_list('organization', flat=True).distinct().count()

        # Son başarının yılı (daha mantıklı bir istatistik)
        latest_achievement_year = None
        if achievements.exists():
            latest_achievement = achievements.exclude(
                date__isnull=True).first()  # ordering = ['-date'] olduğu için first() en yeni
            if latest_achievement and latest_achievement.date:
                latest_achievement_year = latest_achievement.date.year

        context.update({
            'achievements': achievements,
            'featured_count': featured_count,
            'unique_organizations': unique_orgs,
            'latest_achievement_year': latest_achievement_year,
        })

        return context

class FreelanceReviewsView(TemplateView):
    template_name = 'resume/freelancer_platform.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Tüm hizmetleri al ve istatistikleri ekle
        services = FreelanceService.objects.prefetch_related(
            'reviews__platform'
        ).annotate(
            review_count=Count('reviews'),
            avg_communication=Avg('reviews__communication_rating'),
            avg_timing=Avg('reviews__timing_rating'),
            avg_quality=Avg('reviews__quality_rating'),
            avg_overall=Avg('reviews__communication_rating') * 0.33 +
                        Avg('reviews__timing_rating') * 0.33 +
                        Avg('reviews__quality_rating') * 0.34
        ).order_by('-review_count')

        # Öne çıkan yorumlar
        featured_reviews = Review.objects.select_related(
            'service', 'platform'
        ).filter(is_featured=True).order_by('-created_at')[:6]

        # Son yorumlar
        latest_reviews = Review.objects.select_related(
            'service', 'platform'
        ).order_by('-created_at')[:12]

        # Platformlar
        platforms = Platform.objects.all()

        # İstatistikler
        total_reviews = Review.objects.count()
        total_services = FreelanceService.objects.count()
        total_platforms = Platform.objects.count()

        # Genel ortalama puanlar
        if total_reviews > 0:
            avg_ratings = Review.objects.aggregate(
                avg_communication=Avg('communication_rating'),
                avg_timing=Avg('timing_rating'),
                avg_quality=Avg('quality_rating')
            )
            overall_avg = round((
                (avg_ratings['avg_communication'] or 0) +
                (avg_ratings['avg_timing'] or 0) +
                (avg_ratings['avg_quality'] or 0)
            ) / 3, 1)
        else:
            avg_ratings = {
                'avg_communication': 0,
                'avg_timing': 0,
                'avg_quality': 0
            }
            overall_avg = 0

        # En yüksek puanlı hizmet
        top_service = None
        if services.exists():
            top_service = max(services, key=lambda s: s.avg_overall or 0)

        # Context verilerini ekle
        context.update({
            'freelance_services': services,
            'featured_reviews': featured_reviews,
            'latest_reviews': latest_reviews,
            'platforms': platforms,
            'stats': {
                'total_reviews': total_reviews,
                'total_services': total_services,
                'total_platforms': total_platforms,
                'overall_avg': overall_avg,
                'avg_communication': round(avg_ratings['avg_communication'] or 0, 1),
                'avg_timing': round(avg_ratings['avg_timing'] or 0, 1),
                'avg_quality': round(avg_ratings['avg_quality'] or 0, 1),
            },
            'top_service': top_service,
            'star_range': range(1, 6),
        })

        return context

class ContactView(TemplateView):
    template_name = 'resume/contact.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['communications'] = Communication.objects.all()
        return context

    def post(self, request, *args, **kwargs):
        """İletişim formu POST işlemlerini handle eder"""
        try:
            # Form verilerini al
            name = request.POST.get('name', '').strip()
            email = request.POST.get('email', '').strip()
            subject = request.POST.get('subject', '').strip()
            message = request.POST.get('message', '').strip()

            # Basit validasyon
            if not all([name, email, subject, message]):
                messages.error(request, 'All fields are required.')
                return self.get(request, *args, **kwargs)

            # IP adresi ve User Agent bilgilerini al
            ip_address = self.get_client_ip(request)
            user_agent = request.META.get('HTTP_USER_AGENT', '')

            # Veritabanına kaydet
            contact_message = ContactMessage.objects.create(
                name=name,
                email=email,
                subject=subject,
                message=message,
                ip_address=ip_address,
                user_agent=user_agent
            )

            # E-posta gönder
            email_sent = self.send_email_notification(contact_message)

            if email_sent:
                success_message = 'Your message has been sent successfully. I will get back to you as soon as possible.'
                logger.info(f"Contact message sent: {contact_message.id}")
            else:
                success_message = 'Your message has been saved, but the email could not be sent.'
                logger.warning(f"Email could not be sent: {contact_message.id}")

            # AJAX isteği ise JSON response dön
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': success_message
                })

            # Normal request ise mesaj ekle ve redirect et
            if email_sent:
                messages.success(request, success_message)
            else:
                messages.warning(request, success_message)

            return redirect('resume:contact')


        except Exception as e:

            logger.error(f"Contact form error: {str(e)}")

            messages.error(request, 'An error occurred. Please try again.')

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({

                    'success': False,

                    'message': 'An error occurred. Please try again.'

                })

            return self.get(request, *args, **kwargs)

    def get_client_ip(self, request):
        """Kullanıcının gerçek IP adresini alır"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

    def send_email_notification(self, contact_message):
        """E-posta bildirimi gönderir"""
        try:
            # Aktif e-posta ayarlarını al
            email_settings = EmailSettings.get_active_settings()
            if not email_settings or not email_settings.has_password:
                logger.warning("No active email settings found.")
                return False

            # E-posta içeriğini hazırla
            email_context = {
                'contact_message': contact_message,
                'site_name': getattr(settings, 'SITE_NAME', 'Web Sitesi')
            }

            # HTML ve text versiyonları
            html_message = render_to_string('emails/contact_notification.html', email_context)
            plain_message = strip_tags(html_message)

            # E-posta gönder
            from django.core.mail import EmailMultiAlternatives
            from django.core.mail.backends.smtp import EmailBackend

            # Özel backend oluştur
            backend = EmailBackend(
                host=email_settings.smtp_server,
                port=email_settings.smtp_port,
                username=email_settings.email,
                password=email_settings.get_password(),
                use_tls=email_settings.use_tls,
                fail_silently=False,
            )

            subject = f"[İletişim Formu] {contact_message.subject}"
            from_email = email_settings.email
            to_email = [email_settings.email]  # Kendine gönder

            msg = EmailMultiAlternatives(
                subject=subject,
                body=plain_message,
                from_email=from_email,
                to=to_email,
                connection=backend
            )
            msg.attach_alternative(html_message, "text/html")

            # Reply-to header ekle
            msg.reply_to = [contact_message.email]

            msg.send()
            return True

        except Exception as e:
            logger.error(f"Email sending error: {str(e)}")
            return False


def download_resume(request):
    """Özgeçmiş dosyasını indir"""
    language = request.GET.get('lang', 'tr')  # Varsayılan dil Türkçe

    try:
        resume = Resume.objects.filter(
            language=language,
            is_active=True
        ).first()

        if not resume:
            # Eğer istenen dilde özgeçmiş yoksa, varsayılan olarak ilk aktif özgeçmişi al
            resume = Resume.objects.filter(is_active=True).first()

        if not resume:
            raise Http404("Resume not found")

        # Dosya yolu
        file_path = resume.file.path

        if not os.path.exists(file_path):
            raise Http404("File not found")

        # Dosyayı oku ve indir
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/pdf")
            filename = f"CV_{resume.language.upper()}.pdf"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response

    except Exception as e:
        raise Http404("Failed to download the file")

@csrf_exempt
@require_http_methods(["POST"])
def start_chat_session(request):
    """Chat oturumu başlatır"""
    try:
        session_id = str(uuid.uuid4())
        user_ip = request.META.get('REMOTE_ADDR')
        user_agent = request.META.get('HTTP_USER_AGENT')

        session = ChatSession.objects.create(
            session_id=session_id,
            user_ip=user_ip,
            user_agent=user_agent
        )

        # Hoş geldin mesajı
        from .models import Profile  # Profile modelinizin import'u
        profile_name = Profile.objects.first().name if Profile.objects.first() else 'Personal Assistant'
        welcome_message = f"Hello! I am the personal assistant working for {profile_name}. How can I assist you today?"

        ChatMessage.objects.create(
            session=session,
            message_type='bot',
            content=welcome_message
        )

        # Aktif model bilgisini de döndür
        try:
            active_model = GeminiModel.get_active_model()
            model_info = {
                'name': active_model.name,
                'identifier': active_model.model_identifier
            } if active_model else None
        except Exception:
            model_info = None

        return JsonResponse({
            'success': True,
            'session_id': session_id,
            'message': welcome_message,
            'model_info': model_info
        })

    except Exception as e:
        logger.error(f"Chat session initialization error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def rate_limit_check(request, max_requests=10, window=60):
    """Basit rate limiting"""
    user_ip = request.META.get('REMOTE_ADDR')
    cache_key = f'rate_limit_{user_ip}'

    current_requests = cache.get(cache_key, 0)
    if current_requests >= max_requests:
        return False

    cache.set(cache_key, current_requests + 1, window)
    return True

@csrf_exempt
@require_http_methods(["POST"])
def send_chat_message(request):
    """Kullanıcı mesajını işler ve yanıt gönderir"""
    # Rate limiting ekle
    if not rate_limit_check(request):
        return JsonResponse({
            'success': False,
            'error': 'Too many requests. Please wait.'
        }, status=429)

    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        message_content = data.get('message', '').strip()

        if not session_id or not message_content:
            return JsonResponse({
                'success': False,
                'error': 'Session ID and message are required.'
            }, status=400)

        # Session'ı bul
        try:
            session = ChatSession.objects.get(session_id=session_id, is_active=True)
        except ChatSession.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Invalid session.'
            }, status=404)

        # Kullanıcı mesajını kaydet
        user_message = ChatMessage.objects.create(
            session=session,
            message_type='user',
            content=message_content
        )

        try:
            # Context'i al
            context = get_user_context()
            system_prompt = create_system_prompt(context)

            # Select related kullan
            recent_messages = ChatMessage.objects.select_related('session').filter(
                session=session
            ).order_by('-timestamp')[:10]

            # Chat history oluştur
            chat_history = []
            for msg in reversed(recent_messages[1:]):  # Son mesajı hariç tut
                if msg.message_type == 'user':
                    chat_history.append(f"User: {msg.content}")
                elif msg.message_type == 'bot':
                    chat_history.append(f"Assistant: {msg.content}")

            # Prompt oluştur
            full_prompt = f"{system_prompt}\n\nChat History:\n{chr(10).join(chat_history)}\n\nNew User Message: {message_content}\n\nResponse:"

            # Gemini servisi ile yanıt al
            gemini_service = get_gemini_service()
            bot_response, response_time = gemini_service.generate_response(full_prompt, session)

            # Aktif model bilgisini al
            active_model = gemini_service.get_active_model()

            # Bot yanıtını kaydet
            bot_message = ChatMessage.objects.create(
                session=session,
                message_type='bot',
                content=bot_response,
                model_used=active_model.model_identifier,
                response_time=response_time
            )

            # Session'ı güncelle
            session.message_count += 2  # Kullanıcı + bot mesajı
            session.save()

            return JsonResponse({
                'success': True,
                'response': bot_response,
                'response_time': response_time,
                'model_used': {
                    'name': active_model.name,
                    'identifier': active_model.model_identifier
                }
            })

        except Exception as api_error:
            logger.error(f"Gemini API Hatası: {api_error}")

            # Hata durumunda genel yanıt
            fallback_response = "I'm sorry, I'm currently experiencing a technical issue. Please try again later or contact us directly using the provided contact information."
            ChatMessage.objects.create(
                session=session,
                message_type='bot',
                content=fallback_response
            )

            return JsonResponse({
                'success': True,
                'response': fallback_response,
                'error': 'API error'
            })

    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON format'
        }, status=400)
    except Exception as e:
        logger.error(f"Chat message error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def end_chat_session(request):
    """Chat oturumunu sonlandırır"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')

        if not session_id:
            return JsonResponse({
                'success': False,
                'error': 'Session ID required'
            }, status=400)

        try:
            session = ChatSession.objects.get(session_id=session_id)
            session.is_active = False
            session.ended_at = timezone.now()
            session.save()

            return JsonResponse({
                'success': True,
                'message': 'Session ended'
            })

        except ChatSession.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Session not found'
            }, status=404)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def validate_chat_session(request):
    """Session'ın geçerli olup olmadığını kontrol eder"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')

        if not session_id:
            return JsonResponse({'valid': False})

        try:
            session = ChatSession.objects.get(session_id=session_id, is_active=True)

            # Session'ın süresi doldu mu kontrol et (24 saat)
            from django.utils import timezone
            from datetime import timedelta

            expiry_time = timezone.now() - timedelta(hours=24)
            if session.started_at < expiry_time:
                session.is_active = False
                session.ended_at = timezone.now()
                session.save()
                return JsonResponse({'valid': False})

            return JsonResponse({'valid': True})

        except ChatSession.DoesNotExist:
            return JsonResponse({'valid': False})

    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return JsonResponse({'valid': False})

@csrf_exempt
@require_http_methods(["POST"])
def chat_feedback(request):
    """Chat geri bildirimi alır"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        rating = data.get('rating')
        feedback_text = data.get('feedback', '')

        if not session_id:
            return JsonResponse({
                'success': False,
                'error': 'Session ID required'
            }, status=400)

        try:
            session = ChatSession.objects.get(session_id=session_id)

            ChatFeedback.objects.create(
                session=session,
                rating=rating,
                feedback_text=feedback_text
            )

            return JsonResponse({
                'success': True,
                'message': 'Your feedback has been received'
            })

        except ChatSession.DoesNotExist:
            return JsonResponse({
                'success': False,
                'error': 'Session not found'
            }, status=404)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_chat_context(request):
    """Chat context bilgilerini döner (debug için)"""
    try:
        context = get_user_context()

        # Aktif model bilgisini de ekle
        try:
            active_model = GeminiModel.get_active_model()
            model_info = {
                'id': active_model.id,
                'name': active_model.name,
                'identifier': active_model.model_identifier,
                'temperature': active_model.temperature,
                'max_tokens': active_model.max_tokens
            } if active_model else None
        except Exception as e:
            model_info = {'error': str(e)}

        return JsonResponse({
            'success': True,
            'context': context,
            'active_model': model_info
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


# Admin yardımcı görünümleri
@require_http_methods(["POST"])
def refresh_gemini_model(request):
    """Admin panelden model değiştirildiğinde cache'i yenile"""
    if not request.user.is_staff:
        return JsonResponse({
            'success': False,
            'error': 'You do not have the authority'
        }, status=403)

    try:
        from .utils import refresh_gemini_model
        model = refresh_gemini_model()

        return JsonResponse({
            'success': True,
            'message': 'Model cache has been updated',
            'active_model': {
                'name': model.name,
                'identifier': model.model_identifier
            } if model else None
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_model_stats(request):
    """Model istatistiklerini al"""
    if not request.user.is_staff:
        return JsonResponse({
            'success': False,
            'error': 'You do not have the authority'
        }, status=403)

    try:
        from .utils import ModelMetrics

        model_id = request.GET.get('model_id')
        days = int(request.GET.get('days', 7))

        stats = ModelMetrics.get_model_stats(model_id, days)

        return JsonResponse({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


# Test endpoint'i
@require_http_methods(["POST"])
def test_gemini_connection(request):
    """Gemini bağlantısını test et"""
    if not request.user.is_staff:
        return JsonResponse({
            'success': False,
            'error': 'You do not have the authority'
        }, status=403)

    try:
        gemini_service = get_gemini_service()
        test_prompt = "Hello, this is a test message. Please respond briefly."

        response, response_time = gemini_service.generate_response(test_prompt)

        return JsonResponse({
            'success': True,
            'message': 'Connection successful',
            'test_response': response,
            'response_time': response_time,
            'active_model': {
                'name': gemini_service.get_active_model().name,
                'identifier': gemini_service.get_active_model().model_identifier
            }
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)