# Import Module
import os
import pybboxes as pbx

# # Change the directory
# os.chdir(path)
#
# # Read text File
# predictions = {}

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

def calculate_prec_recall(predictions, actual):
    fileNames = list(predictions.keys())
    false_pos = 0;
    false_neg = 0;
    true_pos = 0;
    for name in fileNames:
        if len(predictions[name]) < len(actual[name]):
            false_neg += (len(actual[name]) - len(predictions[name]))
        elif len(predictions[name]) > len(actual[name]):
            false_pos += (len(predictions[name]) - len(actual[name]))
        true_pos += len(actual[name])
    print(false_pos, false_neg, true_pos)
    print("Total_Prec: ", true_pos/(true_pos + false_pos))
    print("Total_Recall: ", true_pos/(true_pos + false_neg))

# Folder Paths
YTPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\test_prec_YT"
YTAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\YoutubeData"
FishPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\test_prec_results"
FishAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\Fish.v1-416x416.darknet"
TotalPred = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\total_pred"
TotalAcc = r"C:\Users\hnvul\Downloads\precision_from_txts\precision_from_txts\total_acc"

predictions, actual = get_pred_acc_dicts(YTPred, YTAcc)
calculate_prec_recall(predictions, actual)

predictions, actual = get_pred_acc_dicts(FishPred, FishAcc)
calculate_prec_recall(predictions, actual)

predictions, actual = get_pred_acc_dicts(TotalPred, TotalAcc)
calculate_prec_recall(predictions, actual)

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
