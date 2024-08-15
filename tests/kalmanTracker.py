import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

def initialize_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.zeros((4, 1), dtype=np.float32)
    return kf

def predict_future_position(kf, steps):
    future_state = kf.statePost.copy()
    future_covariance = kf.errorCovPost.copy()
    
    for _ in range(steps):
        future_state = np.dot(kf.transitionMatrix, future_state)
        future_covariance = np.dot(kf.transitionMatrix, np.dot(future_covariance, kf.transitionMatrix.T)) + kf.processNoiseCov
    
    return future_state[:2]  # Retorna apenas a posição (x, y)

class KalmanTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalman Tracker")
        
        self.slider_window = tk.Toplevel(root)
        self.slider_window.title("Adjust Prediction Steps")
        self.slider_window.geometry("300x150")
        
        self.kf = initialize_kalman()
        self.steps = tk.IntVar(value=5)

        self.slider_label = ttk.Label(self.slider_window, text="Prediction Steps:")
        self.slider_label.pack(pady=10)

        self.steps_slider = ttk.Scale(self.slider_window, from_=1, to=50, orient="horizontal", variable=self.steps, command=self.update_steps_label)
        self.steps_slider.pack()

        self.steps_value_label = ttk.Label(self.slider_window, text=f"Current Steps: {self.steps.get()}")
        self.steps_value_label.pack(pady=10)
        
        self.cap = cv2.VideoCapture(0)
        self.run_tracker()

    def update_steps_label(self, event):
        self.steps_value_label.config(text=f"Current Steps: {int(self.steps.get())}")

    def run_tracker(self):
        ret, frame = self.cap.read()
        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(max_contour)
                center = np.array([[np.float32(x)], [np.float32(y)]])
                
                self.kf.correct(center)
                predicted_state = self.kf.predict()
                future_position = predict_future_position(self.kf, self.steps.get())

                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                predicted_x, predicted_y = int(predicted_state[0]), int(predicted_state[1])
                cv2.circle(frame, (predicted_x, predicted_y), 5, (255, 0, 0), -1)

                future_x, future_y = int(future_position[0]), int(future_position[1])
                cv2.circle(frame, (future_x, future_y), 5, (0, 255, 0), -1)

            cv2.imshow('Kalman Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.root.destroy()
        else:
            self.root.after(10, self.run_tracker)

if __name__ == "__main__":
    root = tk.Tk()
    app = KalmanTrackerApp(root)
    root.mainloop()
