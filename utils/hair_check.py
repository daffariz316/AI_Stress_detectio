import cv2
import numpy as np

def hair_messiness(frame):
    height, width, _ = frame.shape
    roi = frame[0:int(height/4), int(width/4):int(3*width/4)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges > 0) / edges.size

    if edge_density < 0.02:
        status = "Rambut Rapih"
        mess_score = 0
    elif edge_density < 0.05:
        status = "Rambut Agak Acak"
        mess_score = 2
    else:
        status = "Rambut Sangat Acak"
        mess_score = 4

    return mess_score, status
