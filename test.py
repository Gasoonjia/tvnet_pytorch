import cv2
import numpy as np

u1 = np.asarray([0.2309046, 3.72304559e-01])
u2 = np.asarray([0.01136617, -8.99509639e-02])

mag, ang = cv2.cartToPolar(u1, u2)

print(mag, ang)