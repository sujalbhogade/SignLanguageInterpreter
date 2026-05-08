import sys
import cv2
import pyttsx3

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QFrame,
)

from inference import predict
from hand_tracking import hands, get_hand_bbox
from sentence_builder import SentenceBuilder


class ASLApp(QWidget):
    def __init__(self):
        super().__init__()

        # ================= WINDOW =================
        self.setWindowTitle("ASL Sentence Builder")
        self.setMinimumSize(1200, 850)

        # ================= TTS =================
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 140)

        # ================= CAMERA =================
        self.cap = cv2.VideoCapture(0)

        # ================= SENTENCE BUILDER =================
        self.builder = SentenceBuilder()

        # ================= MAIN LAYOUT =================
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        # ================= TITLE =================
        self.title = QLabel("Real-Time Sign Language Recognition")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))

        # ================= CAMERA FRAME =================
        self.camera_frame = QFrame()
        self.camera_frame.setObjectName("cameraFrame")

        camera_layout = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setMinimumHeight(550)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        camera_layout.addWidget(self.image_label)

        self.camera_frame.setLayout(camera_layout)

        # ================= STATUS =================
        self.status_layout = QHBoxLayout()
        self.status_layout.setSpacing(20)

        self.text_label = QLabel("Prediction: --")
        self.text_label.setObjectName("predictionLabel")

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setObjectName("confidenceLabel")

        self.status_layout.addWidget(self.text_label)
        self.status_layout.addWidget(self.confidence_label)

        # ================= SENTENCE BOX =================
        self.sentence_box = QTextEdit()
        self.sentence_box.setReadOnly(True)
        self.sentence_box.setPlaceholderText(
            "Recognized sentence will appear here..."
        )
        self.sentence_box.setMinimumHeight(120)
        self.sentence_box.setObjectName("sentenceBox")

        # ================= BUTTONS =================
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(15)

        self.play_button = QPushButton("▶ Play Sentence")
        self.stop_button = QPushButton("⏹ Stop Speech")
        self.clear_button = QPushButton("🗑 Clear Sentence")

        self.play_button.clicked.connect(self.play_sentence)
        self.stop_button.clicked.connect(self.stop_speech)
        self.clear_button.clicked.connect(self.clear_sentence)

        self.button_layout.addWidget(self.play_button)
        self.button_layout.addWidget(self.stop_button)
        self.button_layout.addWidget(self.clear_button)

        # ================= ADD TO MAIN LAYOUT =================
        self.main_layout.addWidget(self.title)
        self.main_layout.addWidget(self.camera_frame)
        self.main_layout.addLayout(self.status_layout)
        self.main_layout.addWidget(self.sentence_box)
        self.main_layout.addLayout(self.button_layout)

        self.setLayout(self.main_layout)

        # ================= STYLES =================
        self.setStyleSheet("""
            QWidget {
                background-color: #111827;
                color: white;
                font-family: Segoe UI;
            }

            #cameraFrame {
                background-color: #1F2937;
                border-radius: 20px;
                border: 2px solid #374151;
            }

            QLabel {
                color: white;
            }

            #predictionLabel {
                font-size: 22px;
                font-weight: bold;
                color: #60A5FA;
            }

            #confidenceLabel {
                font-size: 22px;
                font-weight: bold;
                color: #34D399;
            }

            #sentenceBox {
                background-color: #1F2937;
                border-radius: 16px;
                border: 2px solid #374151;
                padding: 14px;
                font-size: 22px;
                color: white;
            }

            QPushButton {
                background-color: #2563EB;
                border: none;
                border-radius: 14px;
                padding: 14px;
                font-size: 18px;
                font-weight: bold;
                color: white;
                min-height: 50px;
            }

            QPushButton:hover {
                background-color: #3B82F6;
            }

            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)

        # ================= TIMER =================
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # =========================================================
    # KEY EVENTS
    # =========================================================
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            self.close()

    # =========================================================
    # FRAME UPDATE
    # =========================================================
    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            return

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        hand_results = hands.process(rgb_frame)

        label = "No Hand"
        conf = 0.0

        if hand_results.multi_hand_landmarks:

            hand_landmarks = hand_results.multi_hand_landmarks[0]

            x1, y1, x2, y2 = get_hand_bbox(
                hand_landmarks,
                w,
                h
            )

            hand_crop = frame[y1:y2, x1:x2]

            if (
                hand_crop.size > 0 and
                (x2 - x1) > 50 and
                (y2 - y1) > 50
            ):

                # ================= MODEL PREDICTION =================
                label, conf = predict(hand_crop)

                # ================= SENTENCE BUILDER =================
                label, sentence = self.builder.update(
                    label,
                    conf
                )

                # ================= UPDATE SENTENCE BOX =================
                self.sentence_box.setPlainText(sentence)

                # ================= DRAW BOX =================
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # ================= DRAW LABEL =================
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        # ================= STATUS LABELS =================
        self.text_label.setText(
            f"Prediction: {label}"
        )

        self.confidence_label.setText(
            f"Confidence: {conf:.2f}"
        )

        # ================= QT IMAGE =================
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        bytes_per_line = 3 * w

        qt_img = QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_img)

        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # =========================================================
    # TTS
    # =========================================================
    def play_sentence(self):
        text = self.sentence_box.toPlainText().strip()

        if text:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def stop_speech(self):
        self.tts_engine.stop()

    # =========================================================
    # CLEAR
    # =========================================================
    def clear_sentence(self):
        self.builder.current_sentence = ""
        self.sentence_box.clear()

    # =========================================================
    # CLEANUP
    # =========================================================
    def closeEvent(self, event):
        self.timer.stop()

        if self.cap.isOpened():
            self.cap.release()

        hands.close()

        cv2.destroyAllWindows()

        self.tts_engine.stop()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ASLApp()
    window.show()

    sys.exit(app.exec())
