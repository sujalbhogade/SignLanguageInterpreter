import sys
import cv2
from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout
)

from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer

from inference import predict
from hand_tracking import hands, get_hand_bbox
from sentence_builder import SentenceBuilder

class ASLApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASL Sentence Builder")
        self.setGeometry(100, 100, 800, 700)

        self.image_label = QLabel()

        self.text_label = QLabel("Prediction:")
        self.text_label.setStyleSheet("font-size: 20px")

        self.sentence_label = QLabel("Sentence:")
        self.sentence_label.setStyleSheet(
            "font-size: 28px; font-weight: bold"
        )

        layout = QVBoxLayout()

        layout.addWidget(self.image_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.sentence_label)

        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)

        self.builder = SentenceBuilder()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            self.close()
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

                label, conf = predict(hand_crop)

                label, sentence = self.builder.update(
                    label,
                    conf
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                self.sentence_label.setText(
                    f"Sentence: {sentence}"
                )

        self.text_label.setText(
            f"Prediction: {label} ({conf:.2f})"
        )

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

        self.image_label.setPixmap(
            QPixmap.fromImage(qt_img)
        )

    def closeEvent(self, event):
        self.timer.stop()

        if self.cap.isOpened():
            self.cap.release()

        hands.close()

        cv2.destroyAllWindows()

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ASLApp()
    window.show()

    sys.exit(app.exec())
