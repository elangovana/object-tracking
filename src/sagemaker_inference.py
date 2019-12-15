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
import glob
import os

from predict import Predict

"""
These are sagemaker compatible functions for inference
"""


def input_fn(request_body, request_content_type):
    """An input_fn that processes the request body to a tensor"""
    if request_content_type == 'application/binary':
        return request_body
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise "Unsupported content type {}".format(request_content_type)


def model_fn(model_dir):
    """
    Loads the model from disk and return a model object
    :param model_dir: The directory in which the model is located
    :return: Model object
    """
    model_path = list(glob.glob("{}\*.pth".format(os.path.join(model_dir))))[0]
    num_classes = os.environ.get("NUM_CLASSES", None)
    model_factory_name = os.environ.get("MODEL_FACTORY_NAME", None)

    # Validate to make sure that the model and num classes are available as environment variables
    if num_classes is None or model_factory_name is None:
        raise Exception(
            "Please set the environment variable MODEL_FACTORY_NAME to indicate the model factory name {} and NUM_CLASSES to specify the number of classes {}. This setting should match the training values used to train the model".format(
                model_factory_name, num_classes))

    return Predict(model_dict_path=model_path, num_classes=num_classes, model_factory_name=model_factory_name)


def predict_fn(input_data, model):
    """Predict using input and model"""
    return model(input_data)


def output_fn(prediction, content_type):
    """Return prediction formatted according to the content type"""
    return prediction
