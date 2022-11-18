# Import Module
import os
import shutil
import pybboxes as pbx
import csv

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def read_yolo_cords(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        res =[]
        for line in lines:
            strings = line.split()
            dummy = [eval(i) for i in strings]
            res.append(dummy[1:])
        return res

def read_voc_cords(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        res = []
        for line in lines:
            strings = line.split()
            res.append([eval(i) for i in strings])
        return res

def count_lines(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return (text.count("\n"))

def get_pred_acc_dicts(file_path_pred, file_path_acc):
    # iterate through all prediction files
    predictions = {}
    actual = {}

    os.chdir(file_path_pred)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{file_path_pred}\{file}"
            if (len(file) == 10):
                predictions[f"{file[3:10]}"] = read_voc_cords(file_path)
            else: 
                predictions[f"{file}"] = read_voc_cords(file_path)

    # iterate through all accurate files
    os.chdir(file_path_acc)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{file_path_acc}\{file}"
            if(len(file) == 8):
                actual[f"{file}"] = read_yolo_cords(file_path)
            else:
                actual[f"{file}"] = read_voc_cords(file_path)
    return predictions, actual

# Calculates precision, recall and accuracy on the number of fish detections
# that the model made for each image

#TODO: Check how true_neg is calculated and review that stats calculations is correct (should be, but double check)
def number_stats(name, predictions, actual):
    fileNames = list(predictions.keys())
    false_pos = 0;
    false_neg = 0;
    true_pos = 0;
    true_neg = 0;
    total = 0;
    for name in fileNames:
        if len(predictions[name]) < len(actual[name]):
            false_neg += (len(actual[name]) - len(predictions[name]))
        elif len(predictions[name]) > len(actual[name]):
            false_pos += (len(predictions[name]) - len(actual[name]))
        elif len(predictions[name]) == len(actual[name]) and len(actual[name]) != 0:
            true_pos += len(actual[name])
        else:
            true_neg += 1
        total += len(actual[name])
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    accuracy = (true_pos + true_neg)/len(actual)
    print("False Positive: ", false_pos, "False Negative: ", false_neg, "True Positive: ", true_pos, "True Negative: ", true_neg)
    print("Total Precision: ", precision)
    print("Total Recall: ", recall)
    print("Total Accuracy: ", accuracy, "\n")
    return [name, precision, recall, accuracy]

# Calculates accuracy, precision, and recall on whether the model properly
# detected and didn't detect a fish in each image
def detection_stats(overalldir, dir, predictions, actual):
    filesNames = list(predictions.keys())
    false_pos = 0;
    false_neg = 0;
    true_pos = 0;
    true_neg = 0;
    for name in filesNames:
        if len(predictions[name]) > 0 and len(actual[name]) > 0:
            true_pos += 1
        elif len(predictions[name]) > 0 and len(actual[name]) == 0:
            false_pos += 1
        elif len(predictions[name]) == 0 and len(actual[name]) > 0:
            false_neg += 1
        else:
            true_neg += 1
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    accuracy = (true_pos + true_neg)/len(actual)
    print("False Positive: ", false_pos, "False Negative: ", false_neg, "True Positive: ", true_pos, "True Negative: ", true_neg)
    print("Total Precision: ", precision)
    print("Total Recall: ", recall)
    print("Total Accuracy: ", accuracy, "\n")
    return [overalldir, dir, precision, recall, accuracy, true_pos, false_pos, false_neg, true_neg]

# Returns list of image names that contain no fish
def images_without_fish(dir):
    filesNames = list(dir.keys())
    noFishImages = [];
    for name in filesNames:
        if len(dir[name]) == 0:
            noExt = name[:len(name)-4]
            noFishImages.append(noExt)
    return noFishImages

def false_neg_images(predictions, actual, pred_path):
    modelNoFish = images_without_fish(predictions)
    actualNoFish = images_without_fish(actual)
    falseNeg = []
    for imageName in modelNoFish:
        if imageName not in actualNoFish:
            falseNeg.append(imageName)

    false_neg_dir = pred_path + "\\false_negatives"
    os.mkdir(false_neg_dir)
    os.chdir(pred_path)
    for file in os.listdir():
        if (file.endswith(".jpg") or file.endswith(".png")) and file[:len(file)-13] in falseNeg:
            shutil.copyfile(file, false_neg_dir + "\\" + file)

    return falseNeg


# Cleans out all empty images and its associated txt file from the prediction and actual folders
#TODO: Maybe change to run it on one folder instead of both
def clean_empty_fish(file_path_pred, file_path_acc):
    predictions, actual = get_pred_acc_dicts(file_path_pred, file_path_acc)
    noFishImages = images_without_fish(predictions)
    imagesRemoved = []
    count = 0

    os.chdir(file_path_pred)
    for file in os.listdir():
        if (file.endswith(".jpg") or file.endswith(".png")) and file[:len(file)-13] in noFishImages:
            os.remove(file)
            os.remove(file[:len(file)-13] + ".txt")
            imagesRemoved.append(file)
            count += 1;

    os.chdir(file_path_acc)
    for file in os.listdir():
        if (file.endswith(".jpg") or file.endswith(".png")) and file[:len(file)-13] in noFishImages:
            os.remove(file)
            os.remove(file[:len(file)-13] + ".txt")

    print(f"{count} files removed")
    return imagesRemoved

def create_csv():
    header = ['Overall Folder', 'Individual Test', 'Precision', 'Recall', 'Accuracy', 'True Positive', 'False Positive', 'False Negative', 'True Negative']
    path = r"C:\Users\hnvul\Downloads\carysfort_results"
    csvFile = open(path + "\\" + "results.csv", 'w')
    writer = csv.writer(csvFile)
    writer.writerow(header)
    os.chdir(path)
    for dir in os.listdir():
        currFolder = path + "\\" + dir
        for folder in os.listdir(currFolder):
            print(folder)
            predictions, actual = get_pred_acc_dicts(currFolder + "\\" + folder, CaryAcc)
            false_neg_images(predictions, actual, currFolder + "\\" + folder)
            writer.writerow(detection_stats(dir, folder, predictions, actual))
    csvFile.close()

def empty_txt_images(detection_path, original_path):
    predictions = {}
    os.chdir(detection_path)
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{detection_path}\{file}"
            if (len(file) == 10):
                predictions[f"{file[3:10]}"] = read_voc_cords(file_path)
            else: 
                predictions[f"{file}"] = read_voc_cords(file_path)
    noFish = images_without_fish(predictions)
    os.chdir(original_path)
    emptyTxtDir = original_path + "\\empty_txts"
    os.mkdir(emptyTxtDir)
    for file in os.listdir():
        if (file.endswith(".jpg") or file.endswith(".png")) and file[:len(file)-4] in noFish:
            shutil.copyfile(file, emptyTxtDir + "\\" + file)

# CaryPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\caryforst_predictions"
# CaryOrig = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\cary_orig"

# empty_txt_images(CaryPred, CaryOrig)

# Folder Paths
YTPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\test_prec_YT"
YTAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\YoutubeData"
FishPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\test_prec_results"
FishAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\Fish.v1-416x416.darknet"
CaryPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\caryforst_predictions"
CaryAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\caryforst_actual"
TotalPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\total_pred"
TotalAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\total_acc"

            


# print("Youtube Dataset:")
# print("-----------------------------")
# print("Number of Fish")
# predictions, actual = get_pred_acc_dicts(YTPred, YTAcc)
# number_stats(predictions, actual)
# print("Fish Detection")
# detection_stats(predictions, actual)

# print("Darknet Dataset:")
# print("-----------------------------")
# print("Number of Fish")
# predictions, actual = get_pred_acc_dicts(FishPred, FishAcc)
# number_stats(predictions, actual)
# print("Fish Detection")
# detection_stats(predictions, actual)


# print("Caryforst Dataset:")
# print("-----------------------------")
# print("Number of Fish")
# predictions, actual = get_pred_acc_dicts(CaryPred, CaryAcc)
# number_stats(predictions, actual)
# print("Fish Detection:")
# detection_stats(predictions, actual)

# print("All Datasets")
# print("-----------------------------")
# print("Number of Fish")
# predictions, actual = get_pred_acc_dicts(TotalPred, TotalAcc)
# number_stats(predictions, actual)
# print("Fish Detection")
# detection_stats(predictions, actual)

# print(actual[0][1])
#
# for i in actual:
#     for j in i:
#         print(j)
#         actual[i][j] = pbx.convert_bbox(actual[i][j], from_type="yolo", to_type="voc", image_size=(640,320))
# print(actual)
# box_voc = pbx.convert_bbox(yolo-normalized, from_type="yolo", to_type="voc", image_size=(W,H))
# for i in predictions:
#     fp = max(0,predictions[i] -actual[i])
#     fn
#     print(actual[i])
    # fp = predictions[str(i)+".txt"] -actual[str(i)+".txt"]
