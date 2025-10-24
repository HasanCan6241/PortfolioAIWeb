from django.urls import path
from . import views

app_name = 'resume'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('about/', views.AboutView.as_view(), name='about'),
    path('education/', views.EducationView.as_view(), name='education'),
    path('experience/', views.ExperienceView.as_view(), name='experience'),
    path('skills/', views.SkillsView.as_view(), name='skills'),
    path('projects/', views.ProjectsView.as_view(), name='projects'),
    path('certifications/', views.CertificationsView.as_view(), name='certifications'),
    path('achievements/', views.AchievementsView.as_view(), name='achievements'),
    path('contact/', views.ContactView.as_view(), name='contact'),
    path('download-resume/', views.download_resume, name='download_resume'),
    path('freelance-services/', views.FreelanceReviewsView.as_view(), name='freelance_services'),


    # Chatbot URL'leri
    path('chat/start-session/', views.start_chat_session, name='start_chat_session'),
    path('chat/send-message/', views.send_chat_message, name='send_chat_message'),
    path('chat/end-session/', views.end_chat_session, name='end_chat_session'),
    path('chat/feedback/', views.chat_feedback, name='chat_feedback'),
    path('chat/get-context/', views.get_chat_context, name='get_chat_context'),
    path('validate-session/', views.validate_chat_session, name='validate_chat_session'),

    # Yeni admin/test endpoint'leri
    path('admin/refresh-model/', views.refresh_gemini_model, name='refresh_model'),
    path('admin/model-stats/', views.get_model_stats, name='model_stats'),
    path('admin/test-connection/', views.test_gemini_connection, name='test_connection'),
]