import os
import time
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import cv2 as cv
import csv
from build_testpaths import get_paths
import warnings
from figure_generator import generate_distributions
warnings.filterwarnings("ignore")

ACTIN_TRAIN_DIR = "./actin/train"
ACTIN_TEST_DIR = "./actin/test"

TUBULIN_TRAIN_DIR = "./tubulin/train"
TUBULIN_TEST_DIR = "./tubulin/test"

PARTICIPANT_NAME = 'Fred'  # Change this to your name


# Will be updated within functions
global fps
global fns
global tps
global tns
global fp_files
global fn_files
global index
global labels
global correct
global stats
global start
global end


# Helpers
def false_positive_rate(fp, tn):
    return fp / (fp + tn)


def true_positive_rate(tp, fn):
    return tp / (tp + fn)


def write_results(acc, tposrate, fposrate, numfp, numfn, name=PARTICIPANT_NAME):
    global index
    with open("{}.csv".format(name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = ['Name', 'Accuracy', 'TPR', 'FPR', 'FP', 'FN', 'Classified']
        writer.writerow(header)
        writer.writerow([PARTICIPANT_NAME, round(acc, 3), round(tposrate, 3), round(fposrate, 3),
                        numfp, numfn, index])


def get_test_labels(paths):
    labels = []
    num_normal = 0
    num_anom = 0
    for path in paths:
        fname = path[:-4].split("-")[1]
        score = float(fname)
        if score >= 0.70:
            labels.append(0)
            num_normal += 1
        else:
            labels.append(1)
            num_anom += 1
    return labels, num_normal, num_anom


# setting global vars
test_images, anom_scores = get_paths()
test_labels, num_normal, num_anom = get_test_labels(test_images)

fps = 0
fns = 0
tps = 0
tns = 0
correct = 0
fp_files = []
fn_files = []
index = 0
start = time.time()

labels = test_labels
images = test_images
num_classified = 0
classifications = np.zeros(shape=(len(images), 4))
# number of test_samples * (pred, rating, score, decision_time)


# init tkinter app
win = tk.Tk()
win.geometry('500x500')  # set window size
win.configure(bg='black')

panel = tk.Label(win)
panel.pack()


def next_img():
    global index
    global fps
    global fns
    global tps
    global tns
    global start
    global end
    try:
        # branch if there are still images to display
        print("Image {} of {}".format(index + 1, len(images)))
        img = images[index]  # get the next image from the iterator
    except:
        # branch if reached end of test set
        print("****************************")
        TPR = true_positive_rate(tps, fns)
        FPR = false_positive_rate(fps, tns)
        print("Accuracy: {} / {}, ({}%)".format(correct,
              num_anom + num_normal, 100 * (correct/(num_anom + num_normal))))
        print("True positive rate: {}".format(TPR))
        print("False positive rate: {}".format(FPR))
        print("****************************")

        np.savez('./{}.npz'.format(PARTICIPANT_NAME),
                 classifications=classifications,
                 fp_errors=fp_files,
                 fn_errors=fn_files)

        write_results(correct/(num_anom + num_normal),
                      TPR, FPR, fps, fns)

        exit('\nThank you for your time!')

    # load the image and display
    img_file = np.load(img)
    img = img_file['arr_0']
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
    start = time.time()

# When user presses anomaly


def next_img_anom():
    global index
    global fps
    global fp_files
    global labels
    global images
    global correct
    global tps
    global start
    global end

    end = time.time()
    decision_time = end - start

    if labels[index] != 1:
        fps += 1
        fp_files.append(images[index])
    else:
        tps += 1
        correct += 1
    score = anom_scores[index]
    qual = images[index].split('/')[-1].split('-')[1][:-4]
    classifications[index] = np.array([1, qual, score, decision_time])
    index += 1
    next_img()


# When user presses normal
def next_img_normal():
    global index
    global fns
    global fn_files
    global labels
    global images
    global correct
    global tns
    global start
    global end

    end = time.time()
    decision_time = end - start
    if labels[index] != 0:
        fns += 1
        fn_files.append(images[index])
    else:
        tns += 1
        correct += 1
    score = anom_scores[index]
    qual = images[index].split('/')[-1].split('-')[1][:-4]
    classifications[index] = np.array([0, qual, score, decision_time])
    index += 1
    next_img()


next_img()

# init buttons
btn_normal = tk.Button(
    win, text='Normal', foreground='limegreen', font=('calibri', 16, 'bold'), command=next_img_normal)
btn_normal.pack(expand=True, side=tk.LEFT, fill=tk.BOTH)

btn_anom = tk.Button(win, text='Anomaly',
                     foreground='red', font=('calibri', 16, 'bold'), command=next_img_anom)
btn_anom.pack(expand=True, side=tk.RIGHT, fill=tk.BOTH)


win.mainloop()
