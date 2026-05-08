import time
from collections import deque, Counter

class SentenceBuilder:
    def __init__(self):
        self.pred_buffer = deque(maxlen=5)

        self.current_sentence = ""

        self.confirmed_label = None
        self.stable_frames = 0

        self.required_stable_frames = 25
        self.confidence_threshold = 0.75

        self.last_accept_time = 0
        self.accept_cooldown = 1.0

    def update(self, label, confidence):
        self.pred_buffer.append(label)

        label = Counter(self.pred_buffer).most_common(1)[0][0]

        if confidence < self.confidence_threshold:
            return "...", self.current_sentence

        if label == self.confirmed_label:
            self.stable_frames += 1
        else:
            self.confirmed_label = label
            self.stable_frames = 1

        current_time = time.time()

        if (
            self.stable_frames >= self.required_stable_frames and
            current_time - self.last_accept_time > self.accept_cooldown
        ):

            if label == "space":
                self.current_sentence += " "

            elif label == "del":
                self.current_sentence = self.current_sentence[:-1]

            elif label != "nothing":
                self.current_sentence += label

            self.last_accept_time = current_time
            self.stable_frames = 0

        return label, self.current_sentence
