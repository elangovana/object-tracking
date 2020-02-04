# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
import configparser
import os

import torch

from tracking_datasets.base_tracking_dataset import BaseTrackingDataset

BOXES = "boxes"

IDS = "ids"

FRAMES = "frames"


class Mot17Tracking(BaseTrackingDataset):
    """
    Loads the MOT17 Dataset

    @article{MOT16,
	title = {{MOT}16: {A} Benchmark for Multi-Object Tracking},
	shorttitle = {MOT16},
	url = {http://arxiv.org/abs/1603.00831},
	journal = {arXiv:1603.00831 [cs]},
	author = {Milan, A. and Leal-Taix\'{e}, L. and Reid, I. and Roth, S. and Schindler, K.},
	month = mar,
	year = {2016},
	note = {arXiv: 1603.00831},
	keywords = {Computer Science - Computer Vision and Pattern Recognition}
}


    """

    def __init__(self, root, transform=None):
        super().__init__(root)
        self.transform = transform

        self._samples = self._load_dataset(root)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple:
        sample = self._samples[idx]
        frames = sample[FRAMES]
        boxes = sample[BOXES]
        labels = sample[IDS]

        w, h = sample["w"], sample["h"]

        if self.transform is not None:
            frame, labels["boxes"] = self.transform(frames, w, h, labels["boxes"])

        return frames, boxes, labels

    def _load_dataset(self, root):
        clips = [self._get_clip_details(root, clip_name) for clip_name in os.listdir(root)]

        return clips

    @staticmethod
    def _get_clip_width_height(root, clip_name):
        config = Mot17Tracking._get_config(clip_name, root)
        return float(config["Sequence"]["imWidth"]), float(config["Sequence"]["imHeight"])

    @staticmethod
    def _get_config(clip_name, root):
        seq_ini = os.path.join(root, clip_name, "seqinfo.ini")
        config = configparser.ConfigParser()
        config.read(seq_ini)
        return config

    @staticmethod
    def _get_clip_images_dir(root, clip_name):
        config = Mot17Tracking._get_config(clip_name, root)
        return config["Sequence"]["imDir"]

    @staticmethod
    def _get_clip_images_extension(root, clip_name):
        config = Mot17Tracking._get_config(clip_name, root)
        return config["Sequence"]["imExt"]

    @staticmethod
    def _get_clip_details(root, clip_name) -> dict:
        """

        :rtype: dict
        :param root: The root directory
        :param clip_name: the name of the clip ( the directory name containing the sequence of images for  a single clip and annotations, e.eg MOT17-11-SDP

        returns a dictionary of :
            boxes : the coordinates of the N bounding boxes in [x0, y0, w, h] format, ranging from 0 to W and 0 to H
            frames : the frames in order
            ids : The id of the bounding box, essentially the label
            h: height of the image. All images within the clip r the same height / weight
            w: weight of the image

        """
        gt = os.path.join(root, clip_name, "gt", "gt.txt")

        frame_dict = {}
        # Expected gt file formatted csv with fields <frame>, <id>, <bb left>, <bb top>, <bb width>, <bb height>, <conf>, <class>, <visibility>
        with open(gt, "r") as frame_num:
            for l in frame_num:

                annotations = l.split(",")
                person_class = int(annotations[7])

                # background bounding box, ignore..
                if person_class != 1: continue

                # frame id
                frame = int(annotations[0])

                # Object Id, label
                id = int(annotations[1])

                # bounding box
                px = int(annotations[2])
                py = int(annotations[3])
                pw = int(annotations[4])
                ph = int(annotations[5])

                if frame not in frame_dict:
                    frame_dict[frame] = {BOXES: [],
                                         IDS: []}

                annotation_for_frame = frame_dict.get(frame)

                annotation_for_frame[BOXES].append([px, py, px + pw, py + ph])
                annotation_for_frame[IDS].append(id)

        w, h = Mot17Tracking._get_clip_width_height(root, clip_name)

        images_dir = Mot17Tracking._get_clip_images_dir(root, clip_name)
        image_extn = Mot17Tracking._get_clip_images_extension(root, clip_name)

        # Sort by clip sequences
        result_boxes = [[0]] * len(frame_dict)
        result_ids = [[0]] * len(frame_dict)
        result_frame_seq = [[0]] * len(frame_dict)

        result = {BOXES: result_boxes, IDS: result_ids, FRAMES: result_frame_seq, "h": h, "w": w}

        sorted_keys = sorted(frame_dict.keys())
        for i, frame_num in enumerate(sorted_keys):
            result_boxes[i] = torch.tensor(frame_dict[frame_num][BOXES])
            result_ids[i] = torch.tensor(frame_dict[frame_num][IDS])
            image_name = "{:06d}{}".format(frame_num, image_extn)
            result_frame_seq[i] = os.path.join(root, clip_name, images_dir, image_name)

        return result
