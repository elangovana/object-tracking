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


class Predict:
    """
    Runs predictions on a given model
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, data_loader):
        # Model Eval mode
        self.model.eval()

        predictions = []

        # No grad
        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader):
                # Copy to device
                images = list(image.to(self.device) for image in images)

                predicted_batch = self.model(images)

                predictions.extend(predicted_batch)

        return predictions
