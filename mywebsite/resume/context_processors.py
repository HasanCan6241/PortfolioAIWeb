from .models import Communication, Resume, Profile, Service


def global_context(request):
    """
    Tüm template'lerde kullanılacak global context verileri
    """
    return {
        # İletişim bilgileri (sadece public olanlar)
        'communications': Communication.objects.filter(is_public=True).order_by('order'),

        # Aktif CV'ler
        'available_resumes': Resume.objects.filter(is_active=True),

        # Profil bilgileri (genelde tek profil olduğu varsayımıyla)
        'profile': Profile.objects.first(),

        # Aktif hizmetler (footer için)
        'services': Service.objects.filter(is_active=True).order_by('order'),

        # Tüm iletişim bilgileri (admin için gerekebilir)
        'all_communications': Communication.objects.all().order_by('order'),
    }