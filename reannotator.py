from operator import index
import os
import time
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import cv2 as cv
import csv
import warnings
warnings.filterwarnings('ignore')


NAME = 'FRED'

global img_index
global img_updates

npz_file = np.load('./imgs_to_check.npz')
img_paths = npz_file['paths']
img_paths = img_paths[:5]
img_ids = npz_file['ids']


win = tk.Tk()
win.geometry('500x500')
win.configure(bg='black')

panel = tk.Label(win)
panel.pack()

img_updates = np.zeros((len(img_paths), 2))

img_index = 0


def next_img():
    global img_index
    global img_updates
    try:
        img_path = img_paths[img_index]
    except:
        np.save('./{}/reannotations.npy'.format(NAME), img_updates)
        exit('Finished annotation')
    img = np.load(img_path)
    img = img['arr_0']
    img = cv.normalize(
        img, None, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    cm = plt.get_cmap('hot')
    img = cm(img)
    img = cv.normalize(
        img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_16U)
    img = np.uint8(img)
    img = Image.fromarray(img)
    img = img.resize((400, 400))
    img = ImageTk.PhotoImage(img)
    panel.img = img  # keep a reference so it's not garbage collected
    panel['image'] = img
    if img_index > 0:
        rating = img_path.split('/')[-1].split('-')[-1][:-4]
        text.config(text='Old rating: {}'.format(rating))


def save_annotation_enter(event):
    global img_index
    global img_updates
    global img_ids
    new_rat = float(annotation_entry.get())
    index_to_add = int(img_ids[img_index])
    img_updates[img_index, 0] = new_rat
    img_updates[img_index, 1] = index_to_add
    img_index += 1
    annotation_entry.delete(0, tk.END)
    annotation_entry.insert(0, 'Enter new rating')
    next_img()


def save_annotation_click():
    global img_index
    global img_updates
    global img_ids
    new_rat = float(annotation_entry.get())
    index_to_add = int(img_ids[img_index])
    img_updates[img_index, 0] = new_rat
    img_updates[img_index, 1] = index_to_add
    img_index += 1
    annotation_entry.delete(0, tk.END)
    annotation_entry.insert(0, 'Enter new rating')
    next_img()


next_img()


img_path = img_paths[0]
rating = img_path.split('/')[-1].split('-')[-1][:-4]
text = tk.Label(win, text='Old rating: {}'.format(rating))
text.pack()
annotation_entry = tk.Entry(win)
annotation_entry.insert(0, 'Enter new rating')
annotation_entry.pack(expand=True, side='top')
btn = tk.Button(win, text='Next', font=(
    'calibri', 16, 'bold'), command=save_annotation_click)
btn.pack(expand=True, fill='y')
win.bind('<Return>', save_annotation_enter)


win.mainloop()
