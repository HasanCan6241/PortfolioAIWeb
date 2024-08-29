from django.db import models

class Profile(models.Model):
    profile=models.ImageField(upload_to='profile/', blank=True, null=True)
    def __str__(self):
        return "Proil Resmi"

class Education(models.Model):
    school=models.CharField(max_length=50)
    logo=models.ImageField(upload_to='school_logo/',blank=True,null=True)
    title=models.CharField(max_length=100)
    description = models.TextField()
    gpa=models.CharField(max_length=50)
    def __str__(self):
        return self.school

class About(models.Model):
    about=models.TextField()
    def __str__(self):
        return "Abaout"

class Service(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    def __str__(self):
        return self.name
class Skill(models.Model):
    name = models.CharField(max_length=100)
    level = models.CharField(max_length=50, blank=True, null=True)  # İsteğe bağlı alan
    def __str__(self):
        return self.name

class Project(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField()
    link = models.URLField(blank=True, null=True)  # Link boş bırakılabilir
    def __str__(self):
        return self.title

class Certification(models.Model):
    title = models.CharField(max_length=100)
    institution = models.CharField(max_length=100)
    date = models.DateField()
    certificate_link = models.URLField(max_length=200, blank=True, null=True)
    certificate_image = models.ImageField(upload_to='certifications/', blank=True, null=True)
    def __str__(self):
        return self.title

class Experience(models.Model):
    job_title = models.CharField(max_length=100)
    company = models.CharField(max_length=100)
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    description = models.TextField()
    def __str__(self):
        return self.company

class Achievement(models.Model):
    logo=models.ImageField(upload_to='achievements/', blank=True, null=True)
    title=models.CharField(max_length=100)
    description = models.TextField()
    def __str__(self):
        return self.title

class Communication(models.Model):
    title = models.CharField(max_length=50)
    communication_link = models.URLField(max_length=200, blank=True, null=True)
    def __str__(self):
        return self.title

class Resume(models.Model):
    file = models.FileField(upload_to='resumes/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name