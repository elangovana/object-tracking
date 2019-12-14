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

from datasets.base_detection_dataset import BaseDetectionDataset


class Mot17Detection(BaseDetectionDataset):
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

    The format of the labels is
    <frame>, <id>, <bb left>, <bb top>, <bb width>, <bb height>, <conf>, <class>, <visibility>

    For detection only id is -1,  conf value contains the detection confidence.

    Expect the files to be format:



    """

    @property
    def num_classes(self):
        # 1 class (person) + background
        return 1 + 1

    def __init__(self, root, transform=None):
        super().__init__(root)
        self.transform = transform
        extensions = ('jpg',)
        self._full_list_annotation = {}

        self._samples, self._labels, self._image_width_height = self._make_dataset(root, extensions)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple:
        frame = self._samples[idx]
        labels = self._labels[idx]
        w, h = self._image_width_height[idx]

        if self.transform is not None:
            frame, labels["boxes"] = self.transform(frame, w, h, labels["boxes"])

        tensor_labels = {}
        for k, v in labels.items():
            tensor_labels[k] = torch.tensor(v)
        tensor_labels["boxes"] = tensor_labels["boxes"].float()
        return frame, tensor_labels

    def _make_dataset(self, root, valid_extensions):
        images = []
        labels = []
        image_width_height = []
        is_valid_file = lambda x: os.path.splitext(x)[1][1:] in valid_extensions

        for clip_name in sorted(os.listdir(root)):

            # TODO: validate that there is just one image directory in the MOT17 datatset
            image_dir = os.path.join(root, clip_name, "img1")

            w, h = self._get_clip_width_height(root, clip_name)

            # Load files from clip
            for i, image_frame_name in enumerate(sorted(os.listdir(image_dir))):
                path = os.path.join(image_dir, image_frame_name)
                if is_valid_file(path):
                    labels.append(self._get_labels(root, clip_name, image_frame_name))
                    images.append(path)
                    image_width_height.append((w, h))

        return images, labels, image_width_height

    def _get_clip_width_height(self, root, clip_name):
        seq_ini = os.path.join(root, clip_name, "seqinfo.ini")
        config = configparser.ConfigParser()
        config.read(seq_ini)
        return float(config["Sequence"]["imWidth"]), float(config["Sequence"]["imHeight"])

    def _get_labels(self, root, clip_name, image_frame_name):
        # load all labels for clip
        if clip_name not in self._full_list_annotation:
            self._full_list_annotation[clip_name] = self._load_annotations(root, clip_name)

        # Load labels
        image_index = int(os.path.splitext(image_frame_name)[0])
        return self._full_list_annotation[clip_name][image_index]

    @staticmethod
    def _load_annotations(root, clip_name) -> dict:
        """
        Expected the file format to be
                        #  <frame>, <id>, <bb left>, <bb top>, <bb width>, <bb height>, <conf>, <class>, <visibility>
                        :param root: The root directory
                        :param clip_name: the name of the clip ( the directory name containing the sequence of images for  a single clip and annotations, e.eg MOT17-11-SDP

        target:
        Annotation format : dict containing the following fields, so we can make use of built-in vision loss function for detection ,. See https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
            boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
            labels (Int64Tensor[N]): the label for each bounding box
            image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
            area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
            iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.

        """
        gt = os.path.join(root, clip_name, "gt", "gt.txt")
        result = {}
        with open(gt, "r") as f:
            for l in f:

                annotations = l.split(",")
                person_class = int(annotations[7])

                # background bounding box, ignore..
                if person_class != 1: continue

                # frame if
                frame = int(annotations[0])

                # id = annotations[1]
                px = int(annotations[2])
                py = int(annotations[3])
                pw = int(annotations[4])
                ph = int(annotations[5])

                if frame not in result:
                    result[frame] = {"image_id": frame,
                                     "boxes": [],
                                     "labels": [],
                                     "area": [],
                                     "iscrowd": []}

                annotation_for_frame = result.get(frame)

                annotation_for_frame["boxes"].append([px, py, px + pw, py + ph])
                annotation_for_frame["labels"].append(person_class)
                annotation_for_frame["area"].append(pw * ph)
                annotation_for_frame["iscrowd"].append(0)

        return result
