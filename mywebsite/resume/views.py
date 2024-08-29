from django.shortcuts import render
from .models import Service, Skill, Project, Certification, Experience,Resume,About,Profile,Education,Achievement,Communication
import numpy as np
import pickle
import re
import random
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import os
from django.conf import settings
from django.http import JsonResponse

def portfolio_view(request):
    profile=Profile.objects.last()
    services = Service.objects.all()
    skills = Skill.objects.all()
    projects = Project.objects.all()
    certifications = Certification.objects.all()
    experiences = Experience.objects.all()
    resume = Resume.objects.last()  # Son yüklenen özgeçmişi alır
    about = About.objects.first()
    education=Education.objects.all()
    achievement=Achievement.objects.all()
    communication=Communication.objects.all()
    context = {
        'services': services,
        'skills': skills,
        'projects': projects,
        'certifications': certifications,
        'experiences': experiences,
        'resume': resume,
        'about':about,
        'profile':profile,
        'education':education,
        'achievements':achievement,
        'communications':communication,
    }
    return render(request, 'resume/index.html', context)


# Load model and files from the root directory
model = load_model('chatbot_model.h5')

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

with open('hasancan_chatbot.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_up_sentence(sentence):
    sentence = re.sub(r'\d+', '', sentence)  # Remove digits
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
    sentence = sentence.lower()  # Convert to lowercase
    sentence_words = word_tokenize(sentence)
    sentence_words = [w for w in sentence_words if w not in stop_words]
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    if not intents_list:
        return "Üzgünüm, anlayamadım. Daha fazla bilgi verebilir misiniz?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result



def chatbot(request):
    if request.method == 'POST':
        user_message = request.POST.get('message')

        # Model ve JSON dosyasını her seferinde açmaktan kaçın
        json_path = os.path.join(os.getcwd(), 'hasancan_chatbot.json')
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Kullanıcı mesajını tahmin et
        ints = predict_class(user_message, model)
        res = get_response(ints, data)
        if user_message=="":
            res="Sorry, I don't understand. Can you give me more information?"
        # Yanıtı JSON formatında döndür
        return JsonResponse({'response': res if res else "Sorry, I don't understand. Can you give me more information?"})

    # GET isteği durumunda, chatbot arayüzünü döndür
    return render(request, 'resume/index.html')
