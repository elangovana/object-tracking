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
from detection_datasets.base_detection_dataset_factory import BaseDetectionDatasetFactory
from detection_datasets.mot17_detection import Mot17Detection
from image_preprocessor import ImagePreprocessor


class Mot17DetectionFactory(BaseDetectionDatasetFactory):
    """
    Mot17 dataset factory
    """

    def get_dataset(self, image_dir):
        """

        :param image_dir:
        :return:
        """

        preporcessor = ImagePreprocessor(resize_ratio=.75)
        dataset = Mot17Detection(root=image_dir, transform=preporcessor)

        return dataset
