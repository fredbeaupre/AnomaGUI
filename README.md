# AnomaGUI

<img src="https://github.com/fredbeaupre/AnomaGUI/blob/master/anomaguiApp.png" width="400" height="400" margin="auto">

## To launch the app

1. ` git clone https://github.com/fredbeaupre/AnomaGUI.git`
2. ` cd AnomaGUI`
3. `pip install -r requirements.txt`
4. Change the `PARTICIPANT_NAME` variable (line 19) to your name in the `anomagui.py` file.
5. Run `python anomagui.py` to launch the tkinter app. You should see a window like the one displayed in the screenshot above.
6. For every image displayed, select normal if you think it has a quality score of 0.70 or above, anormal otherwise. Refer to the images in the `examples/` folder for examples of images and their corresponding quality scores. The image's score is in the file name.
7. The next image is displayed automatically on classification of the previous one.
8. The program terminates automatically once you have finished classifying all images in the test set, i.e., 549 images. The program will save your stats (accuracy, true positive rate, false positive rate, etc.) in the `<PARTICIPANT_NAME>.csv` file. It also saves a detailed array of your individual classifications and errors in the file `<PARTICiPANT_NAME.npz>`.
