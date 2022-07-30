#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   convert_rolabelimg2dota.py
@Time    :   2022/07/30 14:56:16
@Author  :   lixiang 
@Version :   1.0
@Contact :   lixiang-85@foxmail.com
@License :   MIT LICENSE
@Desc    :   <awaiting description>
'''

from logging import root
import math
import xml.etree.ElementTree as ET
import os
import glob
import sys
import numpy as np
from pathlib import Path

def convert_rolabelimg2dota(xml_path:str) -> None:
    """
    Args: 
        - `xml_path` (str) : path to roLabelImg label file, like /xx/xx.xml
        
    Returns: 
        - `box_points` (list): shape (N, 8 + 1), N is the number of objects, 8 + 1 is \
            `(x1, y1, x2, y2, x3, y3, x4, y4, class_name)`
    """
    
    with open(xml_path) as f:
        tree = ET.parse(f)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        objects = root.iter('object')
        boxes = [] # list of tuple(cz, cy, w, h, angle), angle is in [0-pi)
        for obj in objects:
            if obj.find('type').text == 'robndbox':
                rbox_node = obj.find('robndbox')
                cat = obj.find('name').text
                rbox = dict()
                for key in ['cx', 'cy', 'w', 'h', 'angle']:
                    rbox[key] = float(rbox_node.find(key).text)
                boxes.append(list((*rbox.values(), cat)))
        print(f"bboxes: {boxes}")
        
        box_points = [] # list of box defined with four vertices
        for box in boxes:
            cx, cy, w, h, ag, cat = box
            alpha_w = math.atan(w / h)
            alpha_h = math.atan(h / w)
            d = math.sqrt(w**2 + h**2) / 2 
            if ag > math.pi / 2:
                beta = ag - math.pi / 2 + alpha_w
                if beta <= math.pi / 2:
                    x1, y1 = cx + d * math.cos(beta), cy + d * math.sin(beta)
                    x2, y2 = cx - d * math.cos(beta), cy - d * math.sin(beta)
                elif beta > math.pi / 2:
                    beta = math.pi - beta
                    x1, y1 = cx - d * math.cos(beta), cy + d * math.sin(beta)
                    x2, y2 = cx + d * math.cos(beta), cy - d * math.sin(beta)
                x3, y3 = x1 - h * math.cos(ag - math.pi / 2), y1 - h * math.sin(ag - math.pi / 2)
                x4, y4 = x2 + h * math.cos(ag - math.pi / 2), y2 + h * math.sin(ag - math.pi / 2) 
            elif ag <= math.pi / 2:
                beta = ag + alpha_h
                if beta <= math.pi / 2:
                    x1, y1 = cx + d * math.cos(beta), cy + d * math.sin(beta)
                    x2, y2 = cx - d * math.cos(beta), cy - d * math.sin(beta)
                elif beta > math.pi / 2:
                    beta = math.pi - beta
                    x1, y1 = cx - d * math.cos(beta), cy + d * math.sin(beta)
                    x2, y2 = cx + d * math.cos(beta), cy - d * math.sin(beta)
                x3, y3 = x1 - w * math.cos(ag), y1 - w * math.sin(ag)
                x4, y4 = x2 + w * math.cos(ag), y2 + w * math.sin(ag)
                points = np.array([x1, y1, x3, y3, x2, y2, x4, y4], dtype=np.int32)
                points[0::2] = np.clip(points[0::2], 0, width)
                points[1::2] = np.clip(points[1::2], 0, height)
            box_points.append([*points, cat])
        return box_points
    

def roLabelImg2DOTA(xml_dir):
    """ convert roLabelImg xml format (cx,cy,w,h,angle) annotation to DOTA Dataset text format \
        (x1, y1, x2, y2, x3, y3, x4, y4, class_name)
    
    Args: 
        - xml_dir (str): path to roLabelImg xml annotation files, like `data/xmls`     
    """
    p = os.path.dirname(xml_dir)
    p = Path(p) / "labels"
    p.mkdir(parents=True, exist_ok=True)
    p = str(p)
    xmls = glob.glob(os.path.join(xml_dir, '*.xml'))
    for name in xmls:
        boxes = convert_rolabelimg2dota(name)
        base_name = os.path.splitext(os.path.basename(name))[0]
        with open(os.path.join(p, f"{base_name}.txt"), 'w') as f:
            for box in boxes:
                f.write(f"{' '.join(list(map(lambda x: str(x), box)))}\n")
            
def test():
    import cv2
    xml_dir = "data/xmls/P0217.xml"
    base_name = os.path.splitext(os.path.basename(xml_dir))[0]
    img_path = os.path.join(os.path.dirname(xml_dir), "../", "images",  f"{base_name}.png")
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    boxes = convert_rolabelimg2dota(xml_dir)
    contours = [] 
    for box in boxes:
        contours.append(box[:8])
    boxes = np.array(contours, dtype=np.int32)
    contours = boxes.reshape(-1, 4, 2)
    for i in range(len(contours)):
        cv2.drawContours(img, contours, i, (0, 255, 0), 3)
    cv2.imwrite(xml_dir.replace(".xml", ".png"), img)
    
    
if __name__ == '__main__':
    # test()
    xml_path = sys.argv[1]
    roLabelImg2DOTA(xml_path)
