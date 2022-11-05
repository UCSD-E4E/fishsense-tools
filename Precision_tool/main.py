# Import Module
import os
import pybboxes as pbx

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
def number_stats(predictions, actual):
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
    print("False Positive: ", false_pos, "False Negative: ", false_neg, "True Positive: ", true_pos, "True Negative: ", true_neg)
    print("Total Precision: ", true_pos/(true_pos + false_pos))
    print("Total Recall: ", true_pos/(true_pos + false_neg))
    print("Total Accuracy: ", (true_pos + true_neg)/total, "\n")

# Calculates accuracy, precision, and recall on whether the model properly
# detected and didn't detect a fish in each image
def detection_stats(predictions, actual):
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

    print("False Positive: ", false_pos, "False Negative: ", false_neg, "True Positive: ", true_pos, "True Negative: ", true_neg)
    print("Total Precision: ", true_pos/(true_pos + false_pos))
    print("Total Recall: ", true_pos/(true_pos + false_neg))
    print("Total Accuracy: ", (true_pos + true_neg)/len(actual), "\n")

# Returns images that contain fish
def images_with_fish(actual):
    filesNames = list(actual.keys())
    fishImages = [];
    for name in filesNames:
        if len(actual[name] != 0):
            fishImages.append(name)
    return fishImages


# Folder Paths
YTPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\test_prec_YT"
YTAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\YoutubeData"
FishPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\test_prec_results"
FishAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\Fish.v1-416x416.darknet"
caryPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\caryforst_predictions"
caryAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\caryforst_actual"
TotalPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\total_pred"
TotalAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\total_acc"

print("Youtube Dataset:")
print("-----------------------------")
print("Number of Fish")
predictions, actual = get_pred_acc_dicts(YTPred, YTAcc)
number_stats(predictions, actual)
print("Fish Detection")
detection_stats(predictions, actual)

print("Darknet Dataset:")
print("-----------------------------")
print("Number of Fish")
predictions, actual = get_pred_acc_dicts(FishPred, FishAcc)
number_stats(predictions, actual)
print("Fish Detection")
detection_stats(predictions, actual)

print("Caryforst Dataset:")
print("-----------------------------")
print("Number of Fish")
predictions, actual = get_pred_acc_dicts(caryPred, caryAcc)
number_stats(predictions, actual)
print("Fish Detection:")
detection_stats(predictions, actual)

print("All Datasets")
print("-----------------------------")
print("Number of Fish")
predictions, actual = get_pred_acc_dicts(TotalPred, TotalAcc)
number_stats(predictions, actual)
print("Fish Detection")
detection_stats(predictions, actual)

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
