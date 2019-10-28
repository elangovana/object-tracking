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
import os

from datasets.custom_detection_dataset import CustomDetectionDataset


class Mots17DetectionDataset(CustomDetectionDataset):
    """
    Loads the MOT17 Dataset
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
        extensions = ('jpg',)
        self._full_list_annotation = {}

        self._samples, self._labels = self._make_dataset(root, extensions)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple:
        frame = self._samples[idx]
        labels = self._labels[idx]

        if self.transform is not None:
            frame = self.transform(frame)

        return frame, labels

    def _make_dataset(self, root, valid_extensions):
        images = []
        labels = []
        is_valid_file = lambda x: os.path.splitext(x)[1][1:] in valid_extensions

        for clip_name in sorted(os.listdir(root)):

            # TODO: validate that there is just one image directory in the MOT17 datatset
            image_dir = os.path.join(root, clip_name, "img1")

            # Load files from clip
            clip_frames = []
            for i, image_frame_name in enumerate(sorted(os.listdir(image_dir))):
                path = os.path.join(image_dir, image_frame_name)
                if is_valid_file(path):
                    clip_frames.append(path)
                    labels.append(self._get_labels(root, clip_name, image_frame_name))
                    images.append(clip_frames)

        return images, labels

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
                px = annotations[2]
                py = annotations[3]
                pw = annotations[4]
                ph = annotations[5]

                if frame not in result:
                    result[frame] = {"px": [], "py": [], "pw": [], "ph": []}

                annotation_for_frame = result.get(frame)
                annotation_for_frame["px"].append(px)
                annotation_for_frame["py"].append(py)
                annotation_for_frame["pw"].append(pw)
                annotation_for_frame["ph"].append(ph)

        return result
