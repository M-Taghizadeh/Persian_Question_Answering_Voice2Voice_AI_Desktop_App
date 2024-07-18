from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import sounddevice as sd
import soundfile as sf
from vosk import Model, KaldiRecognizer
import re
import wave
import requests
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QScrollArea
from PyQt5.QtGui import QPixmap, QFont, QMovie
from PyQt5.QtMultimedia import QSound
import sys


def chatbot_get_response(question):
    user_input = question
    user_tfidf = tfidf_vectorizer.transform([user_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    best_match_index = similarities.argmax()
    best_answer = data['answer'][best_match_index]
    best_title = data['title'][best_match_index]
    best_context = data['context'][best_match_index]
    best_question = data['question'][best_match_index]

    return best_answer, best_title, best_context, best_question, similarities[0][similarities.argmax()]


def speech_to_text():
    samplerate = 44100  # نرخ نمونه‌برداری
    duration = 8  # مدت زمان ضبط به ثانیه
    filename = 'output.wav'  # نام فایل خروجی

    print("The AI model is listening...")
    myrecording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # انتظار برای اتمام ضبط
    print("End of listening.")
    sf.write(filename, myrecording, samplerate)
    wf = wave.open("output.wav", "rb")
    model = Model("vosk-model-small-fa-0.5/") 
    rec = KaldiRecognizer(model, wf.getframerate())

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            pass
        else:
            pass

    result = rec.FinalResult()
    pattern = r'"text" : "([^"]+)"'
    match = re.search(pattern, result)
    if match:
        extracted_text = match.group(1)
        return extracted_text
    return "من یک هوش مصنوعی پرسش و پاسخ هستم که با پایتون نوشته شدم"


def extract_first_sentences(text):
    sentences = text.split(".")
    first_four_sentences = sentences[:2]
    result_text = ". ".join(first_four_sentences).strip()
    if text.endswith("."):
        result_text += "."
    return result_text


def tts_persian(text):
    text = extract_first_sentences(text)
    url = f"https://tts.datacula.com/api/tts?text={text}&model_name=amir"
    audio_url = "output.wav"
    headers = {
        'accept': 'application/json'
    }

    print("The AI model is thinking...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open(audio_url, "wb") as file:
            file.write(response.content)
    else:
        print("خطا در درخواست API:", response.status_code, response.text)
    return audio_url


class VoiceAssistant(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("هوش مصنوعی جامع پرسش و پاسخ دفاع مقدس")
        self.setGeometry(800, 300, 1000, 1000)
        self.layout = QVBoxLayout()

        self.setStyleSheet("""
            QWidget {
                background-color: black;
                color: white;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #003399;
                color: white;
                border: 1px solid 7AFE5A;
            }
        """)

        # Avatar
        self.avatar_label = QLabel(self)
        self.layout.addWidget(self.avatar_label)

        # Load GIF
        self.movie = QMovie("imgs/avatar.gif")
        self.avatar_label.setMovie(self.movie)
        self.movie.start()

        # Adjust the size
        self.movie.setScaledSize(self.avatar_label.sizeHint().scaled(1000, 1000, Qt.KeepAspectRatio))


        # Welcome sound
        self.welcome_sound = QSound("voices/welcome.wav")
        self.welcome_sound.play()

        # Microphone button
        self.mic_button = QPushButton("🎙️", self)
        self.mic_button.setStyleSheet("font-size: 60px;")
        self.mic_button.clicked.connect(self.listen)
        self.layout.addWidget(self.mic_button)

        # About us
        self.about_us_button = QPushButton("About Us", self)
        self.about_us_button.clicked.connect(self.show_about_us)
        self.layout.addWidget(self.about_us_button)

        # Listening Label
        self.listening_label = QLabel(self)
        self.listening_label.setPixmap(QPixmap("listening.png").scaled(200, 200))
        self.listening_label.setAlignment(Qt.AlignCenter)
        self.listening_label.setVisible(False)
        self.layout.addWidget(self.listening_label)

        # Thinking Label
        self.thinking_label = QLabel(self)
        self.thinking_label.setPixmap(QPixmap("thinking.png").scaled(200, 200))
        self.thinking_label.setAlignment(Qt.AlignCenter)
        self.thinking_label.setVisible(False)
        self.layout.addWidget(self.thinking_label)

        # Question Label
        self.question_label = QLabel("", self)
        self.layout.addWidget(self.question_label)

        # Answer Label inside a ScrollArea
        self.scroll_area = QScrollArea(self)
        self.answer_label = QLabel("", self)
        self.answer_label.setWordWrap(True)
        self.scroll_area.setWidget(self.answer_label)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # Set font
        font = QFont("Tahoma", 14)
        self.answer_label.setFont(font)

        # Main widget
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)

        # Sound objects
        self.current_sound = None

    def stop_current_sound(self):
        if self.current_sound is not None:
            self.current_sound.stop()
            self.current_sound = None

    def show_listening(self):
        self.listening_label.setVisible(True)
        self.thinking_label.setVisible(False)
        self.question_label.setText("در حال گوش دادن...")

    def show_thinking(self):
        self.thinking_label.setVisible(True)
        self.listening_label.setVisible(False)
        self.question_label.setText("در حال تفکر...")

    def hide_status_labels(self):
        self.listening_label.setVisible(False)
        self.thinking_label.setVisible(False)

    def listen(self):
        self.stop_current_sound()
        self.mic_button.setDisabled(True)
        self.show_listening()
        self.current_sound = QSound("voices/next-question.wav")
        self.current_sound.play()
        QTimer.singleShot(4500, self.process_response)

    def process_response(self):
        input_text = speech_to_text()
        self.question_label.setText(input_text)

        best_answer, best_title, best_context, best_question, max_similarity = chatbot_get_response(question=input_text)
        if max_similarity < 0.1:
            best_context = "چطور میتوانم کمکتان کنم؟ سوال خود را از من بپرسید"

        self.display_text_gradually(best_context)
        audio_url = tts_persian(text=best_context)
        self.current_sound = QSound(audio_url)
        self.current_sound.play()
        QTimer.singleShot(len(extract_first_sentences(best_context)) * 100, self.enable_button)

    def display_text_gradually(self, text):
        self.answer_label.setText(text)
        
    def enable_button(self):
        self.hide_status_labels()
        self.mic_button.setDisabled(False)

    def show_about_us(self):
        self.about_us_label = QLabel("This is a Voice Assistant developed by Bonyad.", self)
        self.layout.addWidget(self.about_us_label)


if __name__ == "__main__":
    data = pd.read_csv('dataset/train_46451.csv')
    documents =  data['title'] + ' ' + data['question'] + ' ' + data['context'] + ' ' + data['answer']
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    app = QApplication(sys.argv)
    window = VoiceAssistant()
    window.show()
    sys.exit(app.exec_())