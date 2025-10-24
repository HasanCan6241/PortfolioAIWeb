# forms.py
from django import forms
from .models import ContactMessage
import re


class ContactForm(forms.ModelForm):
    """İletişim formu"""

    class Meta:
        model = ContactMessage
        fields = ['name', 'email', 'subject', 'message']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Adınız ve soyadınız',
                'required': True
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'E-posta adresiniz',
                'required': True
            }),
            'subject': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Mesaj konusu',
                'required': True
            }),
            'message': forms.Textarea(attrs={
                'class': 'form-control textarea',
                'placeholder': 'Mesajınızı yazın...',
                'rows': 5,
                'required': True
            }),
        }

    def clean_name(self):
        """Ad soyad doğrulama"""
        name = self.cleaned_data.get('name', '').strip()
        if len(name) < 2:
            raise forms.ValidationError("Ad soyad en az 2 karakter olmalıdır.")
        if not re.match(r'^[a-zA-ZçğıöşüÇĞIİÖŞÜ\s]+$', name):
            raise forms.ValidationError("Ad soyad sadece harf içermelidir.")
        return name

    def clean_subject(self):
        """Konu doğrulama"""
        subject = self.cleaned_data.get('subject', '').strip()
        if len(subject) < 5:
            raise forms.ValidationError("Konu en az 5 karakter olmalıdır.")
        return subject

    def clean_message(self):
        """Mesaj doğrulama"""
        message = self.cleaned_data.get('message', '').strip()
        if len(message) < 10:
            raise forms.ValidationError("Mesaj en az 10 karakter olmalıdır.")

        # Basit spam kontrolü
        spam_keywords = ['viagra', 'casino', 'lottery', 'winner', 'congratulations', 'million dollars']
        message_lower = message.lower()
        for keyword in spam_keywords:
            if keyword in message_lower:
                raise forms.ValidationError("Mesajınız spam olarak değerlendirildi.")

        return message