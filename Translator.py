import os
import cv2
import math
import time
import torch
import pytesseract
import numpy as np
import pandas as pd
from pytesseract import Output
import matplotlib.pyplot as plt
from matplotlib import patches
from doctr.io import DocumentFile
from sklearn.cluster import DBSCAN
from doctr.models import ocr_predictor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
#from transformers import LLaMAForSequenceClassification, LLaMALanguageProcessor
image_path = "MangaEnglish4.png"
lama = 'lama-cleaner --model=lama --device=cpu --port=8080'
lama_url = 'http://127.0.0.1:8080'
chrome_url = 'C:\Program Files\Google\Chrome\Application'
doc = DocumentFile.from_images(image_path)
predictor = ocr_predictor(pretrained=True)
result = predictor(doc)
#model = LLaMAForSequenceClassification.from_pretrained("llama-3-base")
#processor = LLaMALanguageProcessor()
from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-de'  # English to German model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape
lst_cntr = []
lst_coord = []
lst_boundbox = []

def translate(text, src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the text
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    # Perform the translation
    translation = model.generate(**tokenized_text)
    # Decode the translated text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text

def txt_recog(lst_cntr,lst_coord,lst_boundbox,image):    
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text = word.value
                    bbox = word.geometry
                    #print(f"Text: {text}, Bounding Box: {bbox}")
                    if isinstance(word.geometry, tuple) and len(word.geometry) == 2:
                        (xmin, ymin), (xmax, ymax) = word.geometry
                        xmin_abs = int(xmin * width)
                        ymin_abs = int(ymin * height)
                        xmax_abs = int(xmax * width)
                        ymax_abs = int(ymax * height)
                        image_crop = image[xmin_abs:xmax_abs,ymin_abs:ymax_abs]
                        #textpy = pytesseract.image_to_string(image_crop)
                        #print("\n Text using pytesseract: ",textpy)
                        #if textpy != None:
                        #cv2.rectangle(image, (xmin_abs, ymin_abs), (xmax_abs, ymax_abs), (255, 255, 255), 2)                         
                        cv2.rectangle(mask, (xmin_abs, ymin_abs), (xmax_abs, ymax_abs), 255, thickness=-1)                        
                        center_coordinates = (int((xmin_abs + xmax_abs) / 2), int((ymin_abs + ymax_abs) / 2))
                        lst_cntr.append(center_coordinates)
                        radius = 3 
                        color = (255, 0, 0)
                        thickness = -1
                        # cv2.circle(image, center_coordinates, radius, color, thickness)
                        lst_coord.append((xmin_abs,ymin_abs,xmax_abs,ymax_abs)) 
                                                      
    inpainted_img = cv2.inpaint(image, mask, 7, cv2.INPAINT_NS)
    cv2.imwrite("mask.png",mask)
    inpainted_img = clustering(lst_cntr,lst_coord,lst_boundbox,image,inpainted_img)    
    plt.title("Normal Image")
    plt.imshow(image)
    plt.show()
    plt.title("Inpainted Image")
    plt.imshow(inpainted_img)
    plt.grid(True)
    plt.show()

def eucld(x1, y1, x2, y2):
    dist = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return dist 
def removeDuplicates(lst):     
    return [t for t in (set(tuple(i) for i in lst))]
def clustering(lst_cntr,lst_coord,lst_boundbox,og_image,inpainted_image):
    df = pd.DataFrame(lst_cntr)
    dbscan = DBSCAN(eps=35, min_samples=2)
    labels = dbscan.fit_predict(df)
    unique_label = max(labels) + 1
    for i in range(len(labels)): 
        if labels[i] == -1:
            labels[i] = unique_label
            unique_label += 1
    lst1 = []
    for x in np.unique(labels):
        z = 0
        lst2 = []
        for y in labels:
            if y == x:
                lst2.append([df[0][z], df[1][z]])
            z += 1
        r1 = sum(p[0] for p in lst2) / len(lst2)
        r2 = sum(p[1] for p in lst2) / len(lst2)
        dist = max(eucld(r1, r2, p[0], p[1]) for p in lst2)
        lst1.append([r1, r2, dist])         
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.gca().invert_yaxis()
    #scatter = plt.scatter(df[0], df[1], c=labels, marker='o')
    # for i in lst1:
    #     circle1 = patches.Circle((i[0], i[1]), radius= i[2] ,color= (1,0,0,0.2))
    #     ax.add_patch(circle1)
    #     ax.axis('equal')  
    lst_boundbox_temp = []
    for x in np.unique(labels):
        lst_temp = []             
        for y in range(len(labels)):
            if x == labels[y]:
                if len(lst_temp) == 0:
                    lst_temp = [lst_coord[y][0],lst_coord[y][1],lst_coord[y][2],lst_coord[y][3]]
                else:
                    lst_temp = [min(lst_coord[y][0],lst_temp[0]),min(lst_coord[y][1],lst_temp[1]),max(lst_coord[y][2],lst_temp[2]),max(lst_coord[y][3],lst_temp[3])]
        lst_boundbox_temp+=[lst_temp]  
    for i in lst_boundbox_temp:
        for j in lst_boundbox_temp:
            if (i[0] in range(j[0],j[2]) or i[2] in range(j[0],j[2])) and (i[1] in range(j[1],j[3]) or i[3] in range(j[1],j[3])):
                j[0] = i[0] = min(i[0],j[0])
                j[1] = i[1] = min(i[1],j[1])
                j[2] = i[2] = max(i[2],j[2])
                j[3] = i[3] = max(i[3],j[3])
    lst_boundbox =  removeDuplicates(lst_boundbox_temp)
    #translated_text = translate("", "en", "de")
    #print(translated_text)
    #os.system("clear")
    for z in lst_boundbox:
        scale_factor = 5
        dummy_image = og_image[z[1]:z[3],z[0]:z[2]]        
        height, width = dummy_image.shape[:2]
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))
        upscaled_image = cv2.resize(dummy_image, new_dimensions, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("upscaled_image.png",upscaled_image)
        doc = DocumentFile.from_images("upscaled_image.png")
        cv2.rectangle(image, (z[0]*scale_factor, z[1]*scale_factor), (z[2]*scale_factor, z[3]*scale_factor), (0, 255, 0), 2)        
        box_text = predictor(doc)
        txt = ""
        for page1 in box_text.pages:
                for block1 in page1.blocks:
                    for line1 in block1.lines:
                        for word1 in line1.words:
                            text1 = word1.value
                            txt+= " "
                            txt+=text1
        print(txt)
        #translated_text = translate(txt, "en", "de")
        #print(translated_text)
        plt.imshow(dummy_image)
        plt.show()
    return inpainted_image
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)


txt_recog(lst_cntr,lst_coord,lst_boundbox,image)
