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
from PIL import Image
from torchvision.transforms import transforms

from base_image_preprocessor import BaseImagePreprocessor


class ImagePreprocessor(BaseImagePreprocessor):

    def __init__(self, resize_ratio=1 / 4):
        self.resize_ratio = resize_ratio

    def __call__(self, image_path, image_width, image_height, boxes):

        # Convert to PIL image to apply transformations
        image = Image.open(image_path)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')

        # Combine all transforms
        transform_pipeline = transforms.Compose([
            # Randomly apply horizontal crop or flip

            # Resize
            transforms.Resize((int(image_height * self.resize_ratio), int(image_width * self.resize_ratio))),

            # Regular stuff
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 # torch image: C X H X W
                                 std=[0.229, 0.224, 0.225])])

        img_tensor = transform_pipeline(image)

        # resize boxes as well
        new_boxes = self._resize_targets(boxes, image_height, image_width)

        return img_tensor, new_boxes

    def _resize_targets(self, boxes, image_height, image_width):
        new_boxes = []
        for b in boxes:
            x, y, w, h = b[0], b[1], b[2], b[3]

            x = x * self.resize_ratio
            y = y * self.resize_ratio
            w = w * self.resize_ratio
            h = h * self.resize_ratio

            new_boxes.append([x, y, w, h])
        return new_boxes
