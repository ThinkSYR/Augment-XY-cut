import json
import cv2
import time
import random as rnd
import numpy as np
from xycut import bbox2points, recursive_xy_cut, vis_polygons_with_index
"""
refer to XYLayoutLM
"""


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("avg time: {}s".format(
            (end - start) / len(args[0]["documents"])))
        return res
    return wrapper


def XYCut_document(docs):
    """
    适用于XFUND格式的文档文本块排序
    for predict and train
    """
    for i in range(len(docs["documents"])):
        every_doc = docs["documents"][i]["document"]
        boxes = []
        for it in every_doc:
            boxes.append(it["box"])
        # xy cut return index of sorted boxes
        res = []
        recursive_xy_cut(np.asarray(boxes).astype(int),
                         np.arange(len(boxes)), res)
        docs["documents"][i]["document"] = [every_doc[j] for j in res]


# @timeit
def AugmentXYCut_document(docs, lx=0.5, ly=0.5, theta=5):
    """
    适用于XFUND格式的文档文本块排序
    only for train
    """
    def augment_box(box):
        vx = rnd.uniform(-lx, lx)
        vy = rnd.uniform(-ly, ly)
        if abs(vx) > lx:
            box[0] += vx*theta
            box[2] += vx*theta
        if abs(vy) > ly:
            box[1] += vy*theta
            box[3] += vy*theta
        return box

    for i in range(len(docs["documents"])):
        every_doc = docs["documents"][i]["document"]
        boxes = []
        for it in every_doc:
            it["box"] = augment_box(it["box"])
            boxes.append(it["box"])
        # xy cut return index of sorted boxes
        res = []
        recursive_xy_cut(np.asarray(boxes).astype(int),
                         np.arange(len(boxes)), res)
        docs["documents"][i]["document"] = [every_doc[j] for j in res]


if __name__ == "__main__":
    with open("/Users/hsy963/project/TempProj/kie_svrd/DataProcess/data/xftask1/t1.train.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    AugmentXYCut_document(docs)
    # total documents: 1503
    # avg time: 0.009954921103126593s
