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
import torchvision
from PIL import Image
from skimage import io
from torchvision.transforms import transforms


class ImagePreprocessor:

    def __init__(self, original_height, original_width, min_img_size_h, min_img_size_w):
        self.min_img_size_w = min_img_size_w
        self.min_img_size_h = min_img_size_h
        self.original_width = original_width
        self.original_height = original_height

    def __call__(self, image_path):
        image = io.imread(image_path)
        # pre-process data
        image = Image.fromarray(image)

        horizontal_crop = torchvision.transforms.RandomCrop((self.original_height / 4, self.original_width),
                                                            padding=None,
                                                            pad_if_needed=False,
                                                            fill=0, padding_mode='constant')
        # horizontal flip
        horizonatal_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)

        # Combine all transforms
        transform_pipeline = transforms.Compose([
            # Randomly apply horizontal crop or flip
            # torchvision.transforms.RandomApply([horizonatal_flip, horizontal_crop], p=0.5),
            horizonatal_flip,
            # Resize
            # Market150 dataset size is 64 width, height is 128, so we maintain the aspect ratio
            transforms.Resize((self.min_img_size_h, self.min_img_size_w)),
            # Regular stuff
            transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 # torch image: C X H X W
                                 std=[0.229, 0.224, 0.225])])
        img_tensor = transform_pipeline(image)
        # Add batch [N, C, H, W]
        # img_tensor = img.unsqueeze(0)

        return img_tensor
