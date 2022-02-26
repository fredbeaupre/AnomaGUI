import os
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import cv2 as cv
import csv

ACTIN_TRAIN_DIR = "./actin/train"
ACTIN_TEST_DIR = "./actin/test"

TUBULIN_TRAIN_DIR = "./tubulin/train"
TUBULIN_TEST_DIR = "./tubulin/test"

PARTICIPANT_NAME = 'Fred'

global fps
global fns
global tps
global tns
global fp_files
global fn_files
global index
global labels
global images
global correct


def false_positive_rate(fp, tn):
    return fp / (fp + tn)


def true_positive_rate(tp, fn):
    return tp / (tp + fn)


def write_results(acc, tposrate, fposrate, numfp, numfn, name=PARTICIPANT_NAME):
    global index
    with open("{}.csv".format(name), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        header = ['Name', 'Accuracy', 'TPR', 'FPR', 'Num FP', 'Num FN']
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


paths1 = [file_name for file_name in os.listdir(
    ACTIN_TEST_DIR) if file_name.lower().endswith('.npz')]
paths1 = [os.path.join(ACTIN_TEST_DIR, p) for p in paths1]
paths2 = [file_name for file_name in os.listdir(
    TUBULIN_TEST_DIR) if file_name.lower().endswith('.npz')]
paths2 = [os.path.join(TUBULIN_TEST_DIR, p) for p in paths2]
test_paths = paths1 + paths2
np.random.shuffle(test_paths)
test_images = test_paths[:20]
test_labels, num_normal, num_anom = get_test_labels(test_images)

fps = 0
fns = 0
tps = 0
tns = 0
correct = 0
fp_files = []
fn_files = []
index = 0
labels = test_labels
images = test_images
num_classified = 0


win = tk.Tk()
win.geometry('500x500')  # set window size

panel = tk.Label(win)
panel.pack()


def next_img():
    global index
    global fps
    global fns
    global tps
    global tns
    try:
        img = images[index]  # get the next image from the iterator
    except:
        print("\nYour results: \n")
        print("Number of false positives: {}".format(fps))
        print("Number of false negatives: {}".format(fns))
        print("Number of correctly labelled samples: {} / {}\n".format(correct,
              num_anom + num_normal))
        print("True positive rate: {}".format(true_positive_rate(tps, fns)))
        print("False positive rate: {}".format(false_positive_rate(fps, tns)))

        np.savetxt('false_positives.csv', np.array(
            fp_files), delimiter=',', fmt='%s')
        np.savetxt('false_negatives.csv', np.array(
            fn_files), delimiter=',', fmt='%s')
        # if there are no more images, do nothing

        write_results(correct/(num_anom + num_normal),
                      true_positive_rate(tps, fns), false_positive_rate(fps, tns), fps, fns)
        exit("Thank you for you time!")

    # load the image and display it
    file_name = img.split('/')[-1]
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


def next_img_anom():
    global index
    global fps
    global fp_files
    global labels
    global images
    global correct
    global tps
    print("\n{}".format(images[index]))
    if labels[index] != 1:
        fps += 1
        fp_files.append(images[index])
    else:
        tps += 1
        correct += 1
    # print("Anomaly")
    # print(images[index][:-4].split("-")[1])
    index += 1
    next_img()


def next_img_normal():
    global index
    global fns
    global fn_files
    global labels
    global images
    global correct
    global tns
    print("\n{}".format(images[index]))
    if labels[index] != 0:
        fns += 1
        fn_files.append(images[index])
    else:
        tns += 1
        correct += 1

    # print("Normal")
    # print(images[index][:-4].split("-")[1])
    index += 1
    next_img()


next_img()


btn_normal = tk.Button(
    win, text='Normal', foreground='lightgreen', font=('calibri', 16, 'bold'), command=next_img_normal)
btn_normal.pack(expand=True, side=tk.LEFT, fill=tk.BOTH)

btn_anom = tk.Button(win, text='Anomaly',
                     foreground='red', font=('calibri', 16, 'bold'), command=next_img_anom)
btn_anom.pack(expand=True, side=tk.RIGHT, fill=tk.BOTH)


win.mainloop()
