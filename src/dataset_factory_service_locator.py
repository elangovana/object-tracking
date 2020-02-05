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
import importlib
import os
import pkgutil

from detection_datasets.base_detection_dataset_factory import BaseDetectionDatasetFactory


class DatasetFactoryServiceLocator:
    """
    General evaluator factory that automatically loads  factories that are subclasses of BaseDetectionDatasetFactory
    """

    def __init__(self):
        # Expect the datset factory is under detection_datasets path under the parent of the __file__
        datasets_base_dir = "detection_datasets"
        base_class = BaseDetectionDatasetFactory

        # search path
        search_path = os.path.join(os.path.dirname(__file__), datasets_base_dir)

        # load subclasses of BaseDetectionDatasetFactory from detection_datasets
        for _, name, _ in pkgutil.iter_modules([search_path]):
            importlib.import_module(datasets_base_dir + "." + name)

        self._class_name_class_dict = {cls.__name__: cls for cls in base_class.__subclasses__()}

    @property
    def factory_names(self):
        """
        Returns the names of subclasses of BaseDetectionDatasetFactory that can be dynamically loaded
        :return:
        """
        return list(self._class_name_class_dict.keys())

    def get_factory(self, class_name):
        """
        Returns a dataset factory object
        :param class_name: The name of the dataset factory class, see property factory_names to obtain valid list of class names
        :return:
        """
        if class_name in self._class_name_class_dict:
            return self._class_name_class_dict[class_name]()
        else:
            raise ModuleNotFoundError("Module should be in {}".format(self.factory_names))
