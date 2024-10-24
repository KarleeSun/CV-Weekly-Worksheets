import scipy as sp
import cv2

path = "Screenshot 2024-10-23 at 11.09.37.png"

image = cv2.imread(path)
frequency = sp.fft.fft(image)
lowpass = sp.signal.butter(5, 0.1, 'low')

filtered = sp.signal.filtfilt(*lowpass, frequency)
cv2.imshow(filtered)