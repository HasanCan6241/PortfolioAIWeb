import google.generativeai as genai
from django.core.cache import cache
from .models import GeminiModel, GeminiModelUsage
import logging
import time
from django.db import models
from .models import (
    Profile, Education, About, Service, Skill, Project,
    Certification, Experience, Achievement, Communication
)
logger = logging.getLogger(__name__)


class GeminiService:
    """Gemini AI servisi için yardımcı sınıf"""

    def __init__(self):
        self._current_model = None
        self._configured_model_id = None

    def get_active_model(self):
        cache_key = 'active_gemini_model'
        model = cache.get(cache_key)

        if not model:
            model = GeminiModel.objects.select_related().filter(
                is_active=True,
                is_enabled=True
            ).first()

            if not model:
                model = GeminiModel.objects.filter(is_enabled=True).first()

            if model:
                # Cache süresi model değişiklik sıklığına göre ayarla
                cache.set(cache_key, model, timeout=300)  # 5 dakika
            else:
                # Hiç model yoksa hata fırlat
                raise Exception(
                    "No active Gemini model found! Please add and activate at least one model from the admin panel." )

        return model

    def configure_model(self, model=None):
        """Modeli konfigüre et"""
        if not model:
            model = self.get_active_model()

        # Eğer aynı model zaten konfigüre edilmişse tekrar yapma
        if self._configured_model_id == model.id and self._current_model:
            return self._current_model

        try:
            # API key'i konfigüre et
            genai.configure(api_key=model.api_key)

            # Modeli oluştur
            self._current_model = genai.GenerativeModel(model.model_identifier)
            self._configured_model_id = model.id

            logger.info(f"Gemini model has been configured: {model.name}")
            return self._current_model

        except Exception as e:
            logger.error(f"Gemini model configuration error: {str(e)}")
            raise Exception(f"Model configuration failed: {str(e)}")

    def generate_response(self, prompt, session=None):
        """Gemini'den yanıt al"""
        model_record = self.get_active_model()
        configured_model = self.configure_model(model_record)

        start_time = time.time()
        success = True
        error_message = None
        response_text = ""
        tokens_used = 0

        try:
            # Generation config
            generation_config = {
                "temperature": model_record.temperature,
                "max_output_tokens": model_record.max_tokens,
            }

            response = configured_model.generate_content(
                prompt,
                generation_config=generation_config
            )

            response_time = time.time() - start_time
            response_text = response.text

            # Token sayısını tahmin et (gerçek API'den alamıyoruz)
            tokens_used = len(prompt.split()) + len(response_text.split())

            logger.info(f"Gemini response received. Duration: {response_time:.2f}s, Tokens used: {tokens_used}")

        except Exception as e:
            success = False
            error_message = str(e)
            response_time = time.time() - start_time
            response_text = self.get_fallback_response()

            logger.error(f"Gemini API error: {error_message}")

        # Kullanım kaydını oluştur
        try:
            GeminiModelUsage.objects.create(
                model=model_record,
                session=session,
                tokens_used=tokens_used,
                response_time=response_time,
                success=success,
                error_message=error_message
            )
        except Exception as e:
            logger.error(f"Failed to create usage record: {str(e)}")

        if not success:
            raise Exception(error_message)

        return response_text, response_time

    def get_fallback_response(self):
        """Hata durumunda varsayılan yanıt"""
        return "I'm sorry, I'm currently experiencing a technical issue. Please try again later or contact us directly using the provided contact details."

    def clear_cache(self):
        """Model cache'ini temizle"""
        cache.delete('active_gemini_model')
        self._current_model = None
        self._configured_model_id = None


# Singleton instance
gemini_service = GeminiService()


def get_gemini_service():
    """Gemini servisini al"""
    return gemini_service


def refresh_gemini_model():
    """Aktif modeli yeniden yükle - admin panelden model değiştirildiğinde kullanılır"""
    service = get_gemini_service()
    service.clear_cache()
    return service.get_active_model()


def get_user_context():
    """Kullanıcı hakkında tüm bilgileri toplar"""
    try:
        # Profil bilgileri
        profile = Profile.objects.first()

        # Hakkında bilgileri
        about = About.objects.first()

        # Eğitim bilgileri
        educations = Education.objects.all().order_by('-start_date')

        # Deneyimler
        experiences = Experience.objects.all().order_by('-start_date')

        # Projeler
        projects = Project.objects.filter(status='completed').order_by('-start_date')[:10]

        # Beceriler
        skills = Skill.objects.select_related('category').all()

        # Sertifikalar
        certifications = Certification.objects.filter(is_active=True).order_by('-date')

        # Hizmetler
        services = Service.objects.filter(is_active=True).order_by('order')

        # Başarılar
        achievements = Achievement.objects.all().order_by('-date')

        # İletişim bilgileri
        communications = Communication.objects.filter(is_public=True).order_by('order')

        # Context oluştur
        context = {
            'profile': {
                'name': profile.name if profile else '',
                'title': profile.title if profile else '',
                'location': profile.location if profile else '',
                'phone': profile.phone if profile else '',
                'email': profile.email if profile else '',
            },
            'about': {
                'description': about.about if about else '',
                'short_description': about.short_description if about else '',
                'years_of_experience': about.years_of_experience if about else 0,
            },
            'educations': [
                {
                    'school': edu.school,
                    'title': edu.title,
                    'description': edu.description,
                    'gpa': edu.gpa,
                    'start_date': edu.start_date.strftime('%Y-%m') if edu.start_date else '',
                    'end_date': edu.end_date.strftime('%Y-%m') if edu.end_date else '',
                    'is_current': edu.is_current,
                }
                for edu in educations
            ],
            'experiences': [
                {
                    'job_title': exp.job_title,
                    'company': exp.company,
                    'location': exp.location,
                    'employment_type': exp.get_employment_type_display() if exp.employment_type else '',
                    'start_date': exp.start_date.strftime('%Y-%m'),
                    'end_date': exp.end_date.strftime('%Y-%m') if exp.end_date else '',
                    'is_current': exp.is_current,
                    'description': exp.description,
                    'technologies': ", ".join([tech.name for tech in exp.technologies.all()]),
                    'achievements': exp.achievements,
                }
                for exp in experiences
            ],
            'projects': [
                {
                    'title': proj.title,
                    'description': proj.description,
                    'short_description': proj.short_description,
                    'category': proj.category.name,
                    'technologies': ", ".join([tech.name for tech in proj.technologies.all()]),
                    'link': proj.link,
                    'github_link': proj.github_link,
                    'status': proj.get_status_display(),
                    'start_date': proj.start_date.strftime('%Y-%m') if proj.start_date else '',
                    'end_date': proj.end_date.strftime('%Y-%m') if proj.end_date else '',
                }
                for proj in projects
            ],
            'skills': {}
        }

        # Becerileri kategorilere göre grupla
        for skill in skills:
            category = skill.category.name
            if category not in context['skills']:
                context['skills'][category] = []
            context['skills'][category].append({
                'name': skill.name,
                'level': skill.get_level_display() if skill.level else '',
                'percentage': skill.percentage,
            })

        context.update({
            'certifications': [
                {
                    'title': cert.title,
                    'institution': cert.institution,
                    'date': cert.date.strftime('%Y-%m'),
                    'credential_id': cert.credential_id,
                }
                for cert in certifications
            ],
            'services': [
                {
                    'name': service.name,
                    'description': service.description,
                }
                for service in services
            ],
            'achievements': [
                {
                    'title': ach.title,
                    'description': ach.description,
                    'date': ach.date.strftime('%Y-%m') if ach.date else '',
                    'organization': ach.organization,
                }
                for ach in achievements
            ],
            'communications': [
                {
                    'type': comm.get_type_display(),
                    'title': comm.title,
                    'value': comm.value,
                    'link': comm.communication_link,
                }
                for comm in communications
            ]
        })

        return context

    except Exception as e:
        print(f"Context oluşturma hatası: {e}")
        return {}


def create_system_prompt(context):
    """Gelişmiş ve görsel olarak daha çekici sistem prompt'unu oluşturur"""

    profile = context.get('profile', {})
    about = context.get('about', {})

    # Title and basic introduction
    system_prompt = f"""╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    🤖 PERSONAL ASSISTANT SYSTEM                             ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    You are the intelligent assistant on {profile.get('name', 'a software developer')}'s personal website.

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  🎯 YOUR IDENTITY AND ROLE                                                   │
    └─────────────────────────────────────────────────────────────────────────────┘
    • You are the professional representative of software developer {profile.get('name', '')}
    • You have comprehensive knowledge about their career history, technical skills, projects, and achievements
    • You can provide multilingual support - respond in the same language the user asks in
    • You have a professional, friendly, and helpful personality

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  💬 COMMUNICATION RULES                                                      │
    └─────────────────────────────────────────────────────────────────────────────┘

    🌐 LANGUAGE ADAPTATION:
      ✓ Your native language is English
      ✓ If asked in Turkish, respond in Turkish
      ✓ If asked in English, respond in English  
      ✓ For other languages, respond in that language if possible, otherwise in English
      ✓ Adapt language level to user's level (technical/casual conversation)

    💡 RESPONSE STYLE:
      ✓ Organize and format your responses visually
      ✓ Use appropriate emojis and symbols
      ✓ Structure with headings, subheadings, and lists
      ✓ Use proper formatting for code examples
      ✓ Give direct, clear answers for short and concise questions
      ✓ Provide comprehensive explanations for detailed questions
      ✓ Use appropriate terminology for technical topics
      ✓ Always be polite and professional

    🎨 OUTPUT FORMATTING RULES:
      ✓ Emphasize important information with **bold** text
      ✓ Use ## or ### for headings
      ✓ Use • or numbered lists for lists
      ✓ Use ```language format for code blocks
      ✓ Enrich responses visually with emojis
      ✓ Use section headings for long responses
      ✓ Use table format when tables are appropriate
      ✓ Use proper formatting for links and references
      ✓ Use numbered lists for step-by-step explanations

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  🚀 YOUR EXPERTISE AND RESPONSE FORMATS                                     │
    └─────────────────────────────────────────────────────────────────────────────┘
    1. 🔧 **Technical Consulting**: Projects, technologies used, technical approaches
       - Use syntax highlighting for code examples
       - Adopt step-by-step approach in technical explanations

    2. 📈 **Career Guidance**: Experience, educational background, professional development
       - Present information in timeline format
       - Present achievements in a highlighting manner

    3. 🎨 **Project Details**: Developed applications, features, achievements
       - Organize in project card format
       - Create visual hierarchy

    4. 🤝 **Communication Bridge**: Facilitating collaboration and communication opportunities
       - Present contact information in organized format
       - Clearly specify call-to-actions

    🎯 **SPECIAL RESPONSE FORMATTING EXAMPLES:**

    **For Project Introduction:**
    ```
    🎨 **Project Name**
    📝 Short description here
    ⚡ **Technologies:** React, Node.js, MongoDB
    🌐 **Demo:** [link]
    📁 **Source Code:** [github link]
    ```

    **For Skills List:**
    ```
    🛠️ **Category Name**
    • Skill 1 ⭐⭐⭐⭐⭐
    • Skill 2 ⭐⭐⭐⭐
    • Skill 3 ⭐⭐⭐⭐⭐
    ```

    **For Contact Information:**
    ```
    📞 **Let's Get In Touch**
    📧 **Email:** [email]
    💼 **LinkedIn:** [profile]
    📁 **GitHub:** [profile]
    ```

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  ⚡ REMEMBER AND RESPONSE RULES                                              │
    └─────────────────────────────────────────────────────────────────────────────┘
    • Only respond with the real information provided
    • Don't make assumptions about unknown topics  
    • Don't share personal/private information
    • Politely redirect off-topic questions
    • Focus on adding value in every response

    🎨 **IN EVERY RESPONSE:**
    • Use appropriate titles and subtitles
    • Categorize information
    • Add visual elements (emojis, symbols)
    • Create clear paragraph structure
    • Format with readability focus
    • Give examples when necessary
    • Present relevant links in organized manner

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                        👤 PROFESSIONAL INFORMATION                          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    🏷️  NAME: {profile.get('name', 'Not specified')}
    💼 POSITION: {profile.get('title', 'Software Developer')}
    📍 LOCATION: {profile.get('location', 'Not specified')}
    ⏱️  EXPERIENCE: {about.get('years_of_experience', 0)} years
    📧 CONTACT: {profile.get('email', 'Not specified')}

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  📝 ABOUT                                                                   │
    └─────────────────────────────────────────────────────────────────────────────┘
    {about.get('description', 'Professional specialized in software development.')}

    💫 SHORT INTRODUCTION:
    {about.get('short_description', '')}

    """

    # Adding education information
    educations = context.get('educations', [])
    if educations:
        system_prompt += """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  🎓 EDUCATIONAL BACKGROUND                                                   │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
        for edu in educations:
            duration = f"{edu['start_date']} - {edu['end_date'] if edu['end_date'] else 'Ongoing'}"
            gpa_info = f" (📊 GPA: {edu['gpa']})" if edu.get('gpa') else ""
            system_prompt += f"🏛️  {edu['title']} - {edu['school']} ({duration}){gpa_info}\n"
            if edu.get('description'):
                system_prompt += f"   💭 {edu['description']}\n"

    # Adding experience information
    experiences = context.get('experiences', [])
    if experiences:
        system_prompt += """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  💼 PROFESSIONAL EXPERIENCE                                                  │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
        for exp in experiences:
            duration = f"{exp['start_date']} - {exp['end_date'] if exp['end_date'] else 'Present'}"
            employment = f" ({exp['employment_type']})" if exp.get('employment_type') else ""
            location = f" - {exp['location']}" if exp.get('location') else ""
            system_prompt += f"🏢 {exp['job_title']} @ {exp['company']} ({duration}){employment}{location}\n"

            if exp.get('description'):
                system_prompt += f"   📄 Description: {exp['description']}\n"
            if exp.get('technologies'):
                system_prompt += f"   ⚡ Technologies: {exp['technologies']}\n"
            if exp.get('achievements'):
                system_prompt += f"   🏆 Achievements: {exp['achievements']}\n"

    # Adding skills information
    skills = context.get('skills', {})
    if skills:
        system_prompt += """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  🛠️  TECHNICAL SKILLS                                                        │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
        skill_icons = {
            'Programming Languages': '🔤',
            'Frontend': '🎨',
            'Backend': '⚙️',
            'Database': '🗄️',
            'DevOps': '🔧',
            'Mobile': '📱',
            'Tools': '🛠️',
            'Other': '💡'
        }

        for category, category_skills in skills.items():
            icon = skill_icons.get(category, '🔹')
            skill_list = []
            for skill in category_skills:
                skill_info = skill['name']
                if skill.get('level'):
                    level_icon = {'Beginner': '🟢', 'Intermediate': '🟡', 'Advanced': '🔴', 'Expert': '⭐'}.get(
                        skill['level'], '')
                    skill_info += f" {level_icon}({skill['level']})"
                if skill.get('percentage'):
                    progress_bar = '█' * (skill['percentage'] // 10) + '░' * (10 - skill['percentage'] // 10)
                    skill_info += f" [{progress_bar}] {skill['percentage']}%"
                skill_list.append(skill_info)
            system_prompt += f"{icon} {category}:\n   {chr(10).join(f'  • {skill}' for skill in skill_list)}\n"

    # Adding project information
    projects = context.get('projects', [])
    if projects:
        system_prompt += """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  🎯 KEY PROJECTS                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
        for proj in projects:
            duration = f" ({proj['start_date']} - {proj['end_date']})" if proj['start_date'] else ""
            category_icon = {'Web': '🌐', 'Mobile': '📱', 'Desktop': '💻', 'Game': '🎮', 'AI/ML': '🤖'}.get(proj['category'],
                                                                                                       '🔹')
            system_prompt += f"{category_icon} {proj['title']} [{proj['category']}]{duration}\n"
            system_prompt += f"   💭 {proj['short_description']}\n"
            if proj.get('technologies'):
                system_prompt += f"   ⚡ Technologies: {proj['technologies']}\n"
            if proj.get('link'):
                system_prompt += f"   🌐 Demo: {proj['link']}\n"
            if proj.get('github_link'):
                system_prompt += f"   📁 Code: {proj['github_link']}\n"

    # Adding certifications
    certifications = context.get('certifications', [])
    if certifications:
        system_prompt += """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  🏆 CERTIFICATIONS                                                           │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
        for cert in certifications:
            credential = f" (🆔 ID: {cert['credential_id']})" if cert.get('credential_id') else ""
            system_prompt += f"🏅 {cert['title']} - {cert['institution']} ({cert['date']}){credential}\n"

    # Adding achievements
    achievements = context.get('achievements', [])
    if achievements:
        system_prompt += """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  🏆 ACHIEVEMENTS AND AWARDS                                                  │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
        for ach in achievements:
            date_org = []
            if ach.get('date'):
                date_org.append(ach['date'])
            if ach.get('organization'):
                date_org.append(ach['organization'])
            date_org_str = f" ({' - '.join(date_org)})" if date_org else ""
            system_prompt += f"🥇 {ach['title']}{date_org_str}\n"
            if ach.get('description'):
                system_prompt += f"   💭 {ach['description']}\n"

    # Adding services
    services = context.get('services', [])
    if services:
        system_prompt += """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  🎯 SERVICES OFFERED                                                         │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
        for service in services:
            system_prompt += f"✨ {service['name']}: {service['description']}\n"

    # Adding contact information
    communications = context.get('communications', [])
    if communications:
        system_prompt += """
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  📞 COMMUNICATION CHANNELS                                                   │
    └─────────────────────────────────────────────────────────────────────────────┘
    """
        comm_icons = {
            'Email': '📧', 'Phone': '📞', 'LinkedIn': '💼', 'GitHub': '📁',
            'Twitter': '🐦', 'Website': '🌐', 'Discord': '💬', 'Telegram': '✈️'
        }

        for comm in communications:
            icon = comm_icons.get(comm['type'], '📌')
            link_info = f" ({comm['link']})" if comm.get('link') else ""
            system_prompt += f"{icon} {comm['type']}: {comm['title']} - {comm['value']}{link_info}\n"

    system_prompt += """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           ✨ READY TO HELP! ✨                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """

    return system_prompt


# Model performans metrikleri
class ModelMetrics:
    """Model performans metrikleri"""

    @staticmethod
    def get_model_stats(model_id=None, days=7):
        """Model istatistiklerini al"""
        from django.utils import timezone
        from datetime import timedelta

        start_date = timezone.now() - timedelta(days=days)

        queryset = GeminiModelUsage.objects.filter(created_at__gte=start_date)

        if model_id:
            queryset = queryset.filter(model_id=model_id)

        total_requests = queryset.count()
        successful_requests = queryset.filter(success=True).count()
        failed_requests = total_requests - successful_requests

        if total_requests > 0:
            success_rate = (successful_requests / total_requests) * 100
            avg_response_time = queryset.aggregate(
                avg_time=models.Avg('response_time')
            )['avg_time'] or 0
            total_tokens = queryset.aggregate(
                total_tokens=models.Sum('tokens_used')
            )['total_tokens'] or 0
        else:
            success_rate = 0
            avg_response_time = 0
            total_tokens = 0

        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': round(success_rate, 2),
            'avg_response_time': round(avg_response_time, 3),
            'total_tokens': total_tokens,
            'days': days
        }