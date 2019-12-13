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
import torch

from models.base_model_factory import BaseModelFactory
from models.faster_rcnn import FasterRCnn


class FasterRcnnFactory(BaseModelFactory):

    def load_model(self, model_path, num_classes):
        model = FasterRCnn(num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        return model

    def get_model(self, num_classes):
        return FasterRCnn(num_classes)
