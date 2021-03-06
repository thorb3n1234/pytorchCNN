# Script to convert yolo annotations to voc format

# Sample format
# <annotation>
#     <folder>_image_fashion</folder>
#     <filename>brooke-cagle-39574.jpg</filename>
#     <size>
#         <width>1200</width>
#         <height>800</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>head</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>549</xmin>
#             <ymin>251</ymin>
#             <xmax>625</xmax>
#             <ymax>335</ymax>
#         </bndbox>
#     </object>
# <annotation>
import os
import xml.etree.cElementTree as ET
from PIL import Image
from xml.etree import ElementTree
import csv

ANNOTATIONS_DIR_PREFIX = "connection_dataset/annotations/"

DESTINATION_DIR = "connection_dataset/data/"

CLASS_MAPPING = {
    '0': 'connection'
    # Add your remaining classes here.
}


def create_root(file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "folder").text = "images"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root


def create_file(file_prefix, width, height, voc_labels):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    tree.write("{}/{}.xml".format(DESTINATION_DIR, file_prefix))


def read_file(file_path, outfile_name):
    file_prefix = file_path.split(".txt")[0]
    file = open(outfile_name, 'a+')
    tree = ElementTree.parse("connection_dataset/annotations/" + str(file_path))
    root = tree.getroot()


    print(int(float(root.find('object/bndbox/xmin').text)))

    csvline = [
        root.find('filename').text,
        str(int(float(root.find("size/width").text))),
        str(int(float(root.find('size/height').text))),
        root.find('object/name').text,
        str(int(float(root.find('object/bndbox/xmin').text))),
        str(int(float(root.find('object/bndbox/ymin').text))),
        str(int(float(root.find('object/bndbox/xmax').text))),
        str(int(float(root.find('object/bndbox/ymax').text)))
    ]
    print(csvline)
    file.write(','.join(csvline))
    file.write('\n')


def start():
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    csv_file_name = "connection_annotations.csv"
    file = open(csv_file_name, 'w')
    header = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    file.write(','.join(header))
    file.write('\n')
    file.close()

    for filename in os.listdir(ANNOTATIONS_DIR_PREFIX):
        if filename.endswith('xml') and str(filename) != "classes.txt":
            read_file(filename, csv_file_name)
        else:
            print("Skipping file: {}".format(filename))


if __name__ == "__main__":
    start()
