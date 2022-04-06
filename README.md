# AnomaGUI

<img src="https://github.com/fredbeaupre/AnomaGUI/blob/master/anomaguiApp.png" width="400" height="400" margin="auto">

## Anomaly Detection Problem setup

We have a dataset in which actin and tubulin STED images have been given a quality rating between 0 and 1. The higher the rating, the higher the quality of the image. We define "normal" images to be those with a rating higher than 0.70, and anomalies to be those with a rating smaller than 0.60. Images falling in the 0.6 to 0.7 range have been removed from both train and test sets. We have trained a ResNet on the normal images, so that it learns features reflecting the nature of high quality images, and can then detect low quality images as anomalies. To better evaluate the performance of the model, we would like to compare its performance to that of a human expert in a simple classification task (for each test image: normal or anomaly). If you would like to participate, follow the steps below. Your help is much appreciated.

## To classifiy

1. ` git clone https://github.com/fredbeaupre/AnomaGUI.git`
2. ` cd AnomaGUI`
3. `pip install -r requirements.txt`
4. Change the `PARTICIPANT_NAME` variable (line 21) to your name in the `anomagui.py` file.
5. Run `python anomagui.py` to launch the tkinter app. You should see a window like the one displayed in the screenshot above.
6. For every image displayed, select normal if you think it has a quality rating of 0.70 or above, anormal otherwise. Refer to the images in the `examples/` folder for examples of images and their corresponding quality ratings. The image's score is in the file name, e.g: `actin_img_<QUALITY RATING>.png`.
7. The next image is displayed automatically on classification of the previous one.
8. The program terminates automatically once you have finished classifying all images in the test set, i.e., 549 images. The program will save your stats (accuracy, true positive rate, false positive rate, etc.) in the `<PARTICIPANT_NAME>.csv` file. It also saves a detailed array of your individual classifications and errors in the file `<PARTICIPANT_NAME.npz>`.

Classifying the whole test set takes about 30 minutes. Be careful not to close the app window before having finished classifying all the images, as then all progress would be lost.

## To reannotate the ambiguous images

<img src="https://github.com/fredbeaupre/AnomaGUI/blob/master/reannotatorApp.png" width="400" height="400" margin="auto">

1. ` git clone https://github.com/fredbeaupre/AnomaGUI.git`
2. ` cd AnomaGUI`
3. `pip install -r requirements.txt`
4. Change the `NAME` variable (line 14) to your name in the `reannotator.py` file.
5. Run `python reannotator.py` to launch the tkinter app. You should see a window like the one displayed in the screenshot above, with the image, its old quality rating, a text box to enter a new rating, and a 'Next' button.
6. For every image displayed, enter the rating you think is appropriate for the quality of the image and click Next, or hit Enter. The next image will be displayed on click/enter.
7. There are 141 ambiguous images to reannotate. The programm terminates automatically once you have finished reannotating all 141 images. The updated ratings for the corresponding image indices are saved in the file `reannotations.npy` in the folder `<YOUR NAME>`.
