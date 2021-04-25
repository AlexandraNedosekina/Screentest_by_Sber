import cv2
import imutils
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
import urllib.request
import numpy as np

def get_changes(img_1, img_2):
    height, width = img_1.shape[:2]
    img_2 = cv2.resize(img_2, (width, height), interpolation=cv2.INTER_AREA)
    
    diff = cv2.absdiff(img_1, img_2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 500:
            continue
    
        cv2.rectangle(img_1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    return img_1

def get_image_from_url(url):
    request = 'http://mini.s-shot.ru/1024x6000/png/?{}'.format(url)
    
    with urllib.request.urlopen(request) as _url:
        response = BytesIO(_url.read())
        file_bytes = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    return img

def get_difference_between_sites(url_1, url_2):
    img_1 = get_image_from_url(url_1)
    img_2 = get_image_from_url(url_2)
    return get_changes(img_1, img_2)