# services.py
from .models import EmailSettings, Profile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailService:
    """E-posta gönderim servisi"""

    @staticmethod
    def _get_modern_email_style():
        """Modern e-posta stilini döndürür"""
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #374151;
                background-color: #f8fafc;
            }

            .email-container {
                max-width: 600px;
                margin: 0 auto;
                background-color: #ffffff;
                border-radius: 12px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 32px 24px;
                text-align: center;
            }

            .header h1 {
                color: #ffffff;
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 8px;
            }

            .header p {
                color: rgba(255, 255, 255, 0.9);
                font-size: 16px;
            }

            .content {
                padding: 32px 24px;
            }

            .message-card {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 24px;
                margin: 24px 0;
            }

            .info-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                margin-bottom: 24px;
            }

            .info-item {
                background-color: #ffffff;
                padding: 16px;
                border-radius: 6px;
                border: 1px solid #e2e8f0;
            }

            .info-label {
                font-size: 12px;
                font-weight: 500;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }

            .info-value {
                font-size: 14px;
                font-weight: 500;
                color: #111827;
            }

            .message-content {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }

            .message-content h3 {
                color: #111827;
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 12px;
            }

            .message-text {
                color: #374151;
                font-size: 14px;
                line-height: 1.7;
                white-space: pre-wrap;
            }

            .footer {
                background-color: #f8fafc;
                padding: 24px;
                text-align: center;
                border-top: 1px solid #e2e8f0;
            }

            .footer p {
                color: #6b7280;
                font-size: 12px;
            }

            .badge {
                display: inline-block;
                background-color: #10b981;
                color: #ffffff;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 500;
                margin-top: 8px;
            }

            .divider {
                height: 1px;
                background-color: #e2e8f0;
                margin: 24px 0;
            }

            @media (max-width: 480px) {
                .email-container {
                    margin: 0;
                    border-radius: 0;
                }

                .info-grid {
                    grid-template-columns: 1fr;
                }

                .header, .content, .footer {
                    padding: 20px 16px;
                }
            }
        </style>
        """

    @staticmethod
    def send_contact_email(contact_message):
        """İletişim mesajını e-posta olarak gönderir"""
        try:
            # Aktif e-posta ayarlarını al
            email_settings = EmailSettings.get_active_settings()
            if not email_settings:
                logger.error("Aktif e-posta ayarları bulunamadı")
                return False, "E-posta ayarları bulunamadı"

            # Profil bilgilerini al
            profile = Profile.objects.first()
            if not profile or not profile.email:
                logger.error("Profil e-posta adresi bulunamadı")
                return False, "Alıcı e-posta adresi bulunamadı"

            # E-posta içeriğini hazırla
            subject = f"🔔 Yeni İletişim Mesajı: {contact_message.subject}"

            # Modern HTML içerik
            html_content = f"""
            <!DOCTYPE html>
            <html lang="tr">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Yeni İletişim Mesajı</title>
                {EmailService._get_modern_email_style()}
            </head>
            <body>
                <div class="email-container">
                    <div class="header">
                        <h1>📧 Yeni İletişim Mesajı</h1>
                        <p>Website üzerinden yeni bir mesaj aldınız</p>
                        <div class="badge">Yeni Mesaj</div>
                    </div>

                    <div class="content">
                        <div class="info-grid">
                            <div class="info-item">
                                <div class="info-label">Gönderen</div>
                                <div class="info-value">{contact_message.name}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">E-posta</div>
                                <div class="info-value">{contact_message.email}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Konu</div>
                                <div class="info-value">{contact_message.subject}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Tarih</div>
                                <div class="info-value">{contact_message.created_at.strftime('%d.%m.%Y - %H:%M')}</div>
                            </div>
                        </div>

                        <div class="message-content">
                            <h3>💬 Mesaj İçeriği</h3>
                            <div class="message-text">{contact_message.message}</div>
                        </div>

                        <div class="divider"></div>

                        <div class="info-item">
                            <div class="info-label">IP Adresi</div>
                            <div class="info-value">{contact_message.ip_address or 'Bilinmiyor'}</div>
                        </div>
                    </div>

                    <div class="footer">
                        <p>Bu e-posta {profile.name if profile and profile.name else 'Website'} iletişim formu tarafından otomatik olarak gönderilmiştir.</p>
                        <p>📅 {datetime.now().strftime('%d.%m.%Y %H:%M')} tarihinde oluşturulmuştur.</p>
                    </div>
                </div>
            </body>
            </html>
            """

            # Geliştirilmiş düz metin içerik
            text_content = f"""
╔══════════════════════════════════════════════════════════════╗
║                    🔔 YENİ İLETİŞİM MESAJI                   ║
╚══════════════════════════════════════════════════════════════╝

👤 Gönderen Bilgileri:
   • Ad Soyad: {contact_message.name}
   • E-posta: {contact_message.email}
   • IP Adresi: {contact_message.ip_address or 'Bilinmiyor'}

📋 Mesaj Detayları:
   • Konu: {contact_message.subject}
   • Tarih: {contact_message.created_at.strftime('%d.%m.%Y - %H:%M')}

💬 Mesaj İçeriği:
{'-' * 60}
{contact_message.message}
{'-' * 60}

Bu mesaj website iletişim formu üzerinden gönderilmiştir.
Mesajı yanıtlamak için doğrudan {contact_message.email} adresini kullanabilirsiniz.

═══════════════════════════════════════════════════════════════
📧 {profile.name if profile and profile.name else 'Website'} - Otomatik Bildirim Sistemi
📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}
            """.strip()

            # E-posta gönder
            success = EmailService._send_email_smtp(
                email_settings=email_settings,
                to_email=profile.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )

            if success:
                logger.info(f"İletişim e-postası gönderildi: {contact_message.email}")
                return True, "E-posta başarıyla gönderildi"
            else:
                return False, "E-posta gönderilemedi"

        except Exception as e:
            logger.error(f"E-posta gönderme hatası: {str(e)}")
            return False, f"E-posta gönderme hatası: {str(e)}"

    @staticmethod
    def _send_email_smtp(email_settings, to_email, subject, html_content, text_content):
        """SMTP ile e-posta gönderimi"""
        try:
            # Şifreyi çöz
            password = email_settings.get_password()
            if not password:
                logger.error("E-posta şifresi çözülemedi")
                return False

            # SMTP bağlantısı
            server = smtplib.SMTP(email_settings.smtp_server, email_settings.smtp_port)

            if email_settings.use_tls:
                server.starttls()

            server.login(email_settings.email, password)

            # E-posta mesajını oluştur
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = email_settings.email
            msg['To'] = to_email

            # İçerikleri ekle
            text_part = MIMEText(text_content, 'plain', 'utf-8')
            html_part = MIMEText(html_content, 'html', 'utf-8')

            msg.attach(text_part)
            msg.attach(html_part)

            # Gönder
            server.send_message(msg)
            server.quit()

            return True

        except Exception as e:
            logger.error(f"SMTP e-posta gönderme hatası: {str(e)}")
            return False

    @staticmethod
    def send_auto_reply(contact_message):
        """Otomatik yanıt gönderimi"""
        try:
            email_settings = EmailSettings.get_active_settings()
            if not email_settings:
                return False, "E-posta ayarları bulunamadı"

            profile = Profile.objects.first()

            subject = f"✅ Mesajınızı aldık - {contact_message.subject}"

            # Modern otomatik yanıt HTML içerik
            html_content = f"""
            <!DOCTYPE html>
            <html lang="tr">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Mesajınızı Aldık</title>
                {EmailService._get_modern_email_style()}
            </head>
            <body>
                <div class="email-container">
                    <div class="header">
                        <h1>✅ Mesajınız Alındı</h1>
                        <p>Size en kısa sürede geri dönüş yapacağız</p>
                        <div class="badge">Otomatik Yanıt</div>
                    </div>

                    <div class="content">
                        <p style="font-size: 16px; margin-bottom: 24px;">Merhaba <strong>{contact_message.name}</strong>,</p>

                        <p style="font-size: 14px; color: #6b7280; margin-bottom: 24px;">
                            Mesajınızı başarıyla aldık ve en kısa sürede size geri dönüş yapacağız. 
                            Genellikle 24 saat içerisinde yanıtlıyoruz.
                        </p>

                        <div class="message-card">
                            <h3 style="color: #111827; font-size: 16px; margin-bottom: 16px;">📋 Mesaj Özeti</h3>
                            <div class="info-grid">
                                <div class="info-item">
                                    <div class="info-label">Konu</div>
                                    <div class="info-value">{contact_message.subject}</div>
                                </div>
                                <div class="info-item">
                                    <div class="info-label">Gönderim Tarihi</div>
                                    <div class="info-value">{contact_message.created_at.strftime('%d.%m.%Y - %H:%M')}</div>
                                </div>
                            </div>
                        </div>

                        <div class="divider"></div>

                        <p style="font-size: 14px; color: #374151;">
                            Teşekkür ederiz! 🙏<br>
                            <strong>{profile.name if profile and profile.name else 'Website Ekibi'}</strong>
                        </p>
                    </div>

                    <div class="footer">
                        <p>Bu otomatik bir yanıttır. Lütfen bu e-postayı yanıtlamayın.</p>
                        <p>📧 {email_settings.email} - 📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
                    </div>
                </div>
            </body>
            </html>
            """

            # Geliştirilmiş otomatik yanıt düz metin
            text_content = f"""
╔══════════════════════════════════════════════════════════════╗
║                    ✅ MESAJINIZ ALINDI                       ║
╚══════════════════════════════════════════════════════════════╝

Merhaba {contact_message.name},

Mesajınızı başarıyla aldık ve en kısa sürede size geri dönüş yapacağız.
Genellikle 24 saat içerisinde yanıtlıyoruz.

📋 Mesaj Detayları:
   • Konu: {contact_message.subject}
   • Gönderim Tarihi: {contact_message.created_at.strftime('%d.%m.%Y - %H:%M')}

Teşekkür ederiz! 🙏

İyi günler,
{profile.name if profile and profile.name else 'Website Ekibi'}

═══════════════════════════════════════════════════════════════
Bu otomatik bir yanıttır. Lütfen bu e-postayı yanıtlamayın.
📧 {email_settings.email} - 📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}
            """.strip()

            success = EmailService._send_email_smtp(
                email_settings=email_settings,
                to_email=contact_message.email,
                subject=subject,
                html_content=html_content,
                text_content=text_content
            )

            return success, "Otomatik yanıt gönderildi" if success else "Otomatik yanıt gönderilemedi"

        except Exception as e:
            logger.error(f"Otomatik yanıt gönderme hatası: {str(e)}")
            return False, str(e)