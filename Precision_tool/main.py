# Import Module
import os
import pybboxes as pbx
# Folder Path
path = r"C:\Users\ragha\Desktop\python_projects\precision_from_txts\test_prec_YT"

# Change the directory
os.chdir(path)

# Read text File
predictions = {}


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


# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}\{file}"
        predictions[f"{file}"] = read_voc_cords(file_path)

# Folder Path
path2 = r"C:\Users\ragha\Desktop\python_projects\precision_from_txts\YoutubeData"

# Change the directory
os.chdir(path2)

# Read text File
actual = {}
# W,H = 640,320
# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path2}\{file}"
        actual[f"{file}"] = read_yolo_cords(file_path)
# print(predictions)
print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
# print(actual[0][1])

for i in actual:
    for j in i:
        print(j)
        actual[i][j] = pbx.convert_xbbox(actual[i][j], from_type="yolo", to_type="voc", image_size=(640,320))
print(actual)
# box_voc = pbx.convert_bbox(yolo-normalized, from_type="yolo", to_type="voc", image_size=(W,H))
# for i in predictions:
#     fp = max(0,predictions[i] -actual[i])
#     fn
#     print(actual[i])
    # fp = predictions[str(i)+".txt"] -actual[str(i)+".txt"]