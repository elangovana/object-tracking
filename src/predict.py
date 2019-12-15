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
import tempfile

import torch
from torchvision import transforms

from model_factory_service_locator import ModelFactoryServiceLocator


class Predict:
    """
    Runs predictions on a given model
    """

    def __init__(self, model_factory_name, model_dict_path, num_classes, device=None):
        self.model_factory_name = model_factory_name
        model_factory = ModelFactoryServiceLocator().get_factory(model_factory_name)
        model = model_factory.load_model(model_dict_path, num_classes)
        self.model = model
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, input_file_or_bytes):
        # If file
        if isinstance(input_file_or_bytes, str):
            input_data = self._pre_process_image(input_file_or_bytes)
        # Else bytes
        elif isinstance(input_file_or_bytes, bytes):
            with tempfile.NamedTemporaryFile("w+b") as f:
                f.write(input_file_or_bytes)
                f.seek(0)
                input_data = self._pre_process_image(f)
        else:
            input_data = input_file_or_bytes

        self.model.eval()

        with torch.no_grad():
            predicted_batch = self.model(input_data)

        return predicted_batch

    def predict_batch(self, data_loader):
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

    def _pre_process_image(self, input_file_or_bytes):
        # Combine all transforms
        transform_pipeline = transforms.Compose([
            # Regular stuff
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 # torch image: C X H X W
                                 std=[0.229, 0.224, 0.225])])

        img_tensor = transform_pipeline(input_file_or_bytes)

        return img_tensor
