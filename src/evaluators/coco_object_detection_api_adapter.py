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

from pycocotools.coco import COCO


class CocoObjectDetectionApiAdapter:
    """
    Converts torch vision object detection object to coco api format, described in http://cocodataset.org/#format-data

    {
        "info" : info,
        "images" : [image],
        "annotations" : [annotation],
        "licenses" : [license],
    }


    info{
    "year" : int,
    "version" : str,
    "description" : str,
    "contributor" : str,
    "url" : str,
    "date_created" : datetime,
    }

    image{
    "id" : int,
    "width" : int,
    "height" : int,
    "file_name" : str,
    "license" : int,
    "flickr_url" : str,
    "coco_url" : str,
     "date_captured" : datetime,
    }

    license{
    "id" : int,
    "name" : str,
    "url" : str,
    }


    annotation{
     "id" : int,
     "image_id" : int,
     "category_id" : int,
     "segmentation" : RLE or [polygon],
     "area" : float,
     "bbox" : [x,y,width,height],
     "iscrowd" : 0 or 1,
    }

    categories[{
    "id" : int, "name" : str, "supercategory" : str,
    }]

    """

    def __init__(self):
        pass

    def to_coco(self, image_anno_boxes_gt, image_anno_boxes_p):
        """
        Coverts torch vision annotation to coco format
        :param image_anno_boxes_p: The predictions from torch vision model
        :param image_anno_boxes_gt:  The groundtruth from torchvision dataset
        :return: coco dataset annotation
        """
        coco_annotations_gt = []
        coco_annotations_p = []

        categories = set()
        images = set()
        for torch_vision_gt_anno, torch_vision_pred in zip(image_anno_boxes_gt, image_anno_boxes_p):
            image_id = torch_vision_gt_anno["image_id"].item()

            annotations_gt = self._get_coco_annotations_for_image(torch_vision_gt_anno, image_id)
            annotations_p = self._get_coco_annotations_for_image(torch_vision_pred, image_id)

            coco_annotations_gt.extend(annotations_gt)
            coco_annotations_p.extend(annotations_p)

            categories = categories.union([a["category_id"] for a in coco_annotations_gt])
            images = images.union([a["image_id"] for a in coco_annotations_gt])

        # Populate the unique categories and the images
        coco_categories = [{"id": c, "name": str(c)} for c in list(categories)]
        coco_images = [{"id": i} for i in list(images)]

        # Convert to COCO gt
        coco_gt = self._build_coco_dataset(coco_annotations_gt, coco_categories, coco_images)

        # Convert to COCO predictions
        coco_p = self._build_coco_dataset(coco_annotations_p, coco_categories, coco_images)

        return coco_gt, coco_p

    def _build_coco_dataset(self, coco_annotations, coco_categories, coco_images):
        self._populate_id(coco_annotations)
        coco_gt = COCO()
        coco_gt.dataset = {
            "annotations": coco_annotations,
            "categories": coco_categories,
            "images": coco_images
        }
        coco_gt.createIndex()
        return coco_gt

    def _get_coco_annotations_for_image(self, torch_vision_anno, image_id):
        anotations = []
        for i, box in enumerate(torch_vision_anno["boxes"]):
            #
            # """
            # annotation{
            #     "id" : int,
            #     "image_id" : int,
            #     "category_id" : int,
            #     "segmentation" : RLE or [polygon],
            #     "area" : float,
            #     "bbox" : [x,y,width,height],
            #     "iscrowd" : 0 or 1,
            #    }
            # """
            #
            category = torch_vision_anno["labels"][i].item()
            box_lst = box.numpy().tolist()

            annotation = {
                "id": None,
                "image_id": image_id,
                "category_id": category,
                "bbox": box_lst,
                'area': box_lst[2] * box_lst[3]
            }

            if "iscrowd" in torch_vision_anno:
                annotation["iscrowd"] = torch_vision_anno["iscrowd"][i].item()

            if "scores" in torch_vision_anno:
                annotation["score"] = torch_vision_anno["scores"][i].item()

            anotations.append(annotation)

        return anotations

    def _populate_id(self, annotations):

        for i, a in enumerate(annotations):
            # The ids should be 1 indexed as the COCO eval doesnt work ..
            a["id"] = i + 1
