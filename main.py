from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt 

img1_path = '/mnt/01D624E6112E18C0/FAISS/deepface/tests/dataset/img1.jpg'
img2_path = '/mnt/01D624E6112E18C0/FAISS/deepface/tests/dataset/img2.jpg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)


result = DeepFace.verify(img1_path, img2_path)
print (result)
