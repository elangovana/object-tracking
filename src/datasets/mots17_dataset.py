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


class Mots17Dataset(CustomDetectionDataset):
    """
    Loads the MOT17 Dataset
    The format of the labels is
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>

    For detection only id is -1,  conf value contains the detection confidence.

    Expect the files to be format:

    """

    @property
    def num_classes(self):
        # 1 class (person) + background
        return 1 + 1

    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips=1,
                 fold=1, train=True, transform=None):
        super().__init__(root)
        extensions = ('jpg',)
        self._samples, self._labels = self._make_dataset(root, extensions)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        clip_name, frames = self._samples[idx]
        label = self._labels[clip_name]

        if self.transform is not None:
            frames = self.transform(frames)

        return frames, label

    def _make_dataset(self, root, valid_extensions):
        images = []
        labels = {}
        is_valid_file = lambda x: os.path.splitext(x)[1][1:] in valid_extensions

        for clip_name in sorted(os.listdir(root)):

            # TODO: validate that there is just one image directory in the MOT17 datatset
            image_dir = os.path.join(root, clip_name, "img1")

            # Load labels
            gt = os.path.join(root, clip_name, "gt", "gt.txt")
            labels[clip_name] = {}
            with open(gt, "r") as f:
                for l in f:
                    # < frame >, < id >, < bb_left >, < bb_top >, < bb_width >, < bb_height >, < conf >
                    annotations = l.split(",")
                    frame = int(annotations[0])
                    id = annotations[1]
                    px = annotations[2]
                    py = annotations[3]
                    pw = annotations[4]
                    ph = annotations[5]

                    if not id in labels[clip_name]:
                        labels[clip_name][id] = {"frames": [], "px": [], "py": [], "pw": [], "ph": []}

                    labels[clip_name][id]["frames"].append(frame)
                    labels[clip_name][id]["px"].append(px)
                    labels[clip_name][id]["py"].append(py)
                    labels[clip_name][id]["pw"].append(pw)
                    labels[clip_name][id]["ph"].append(ph)

            clip_frames = []
            for i, image_frame_name in enumerate(sorted(os.listdir(image_dir))):
                path = os.path.join(image_dir, image_frame_name)
                if is_valid_file(path):
                    image_index = os.path.splitext(image_frame_name)[0]
                    assert int(image_index) == i + 1, "Expect the image name {} to match int {}".format(image_index,
                                                                                                        i + 1)

                    clip_frames.append(path)
            images.append((clip_name, clip_frames))

        return images, labels
