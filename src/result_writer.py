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
import datetime
import json
import logging
import os
import uuid


class ResultWriter:
    """
    Writes results
    """

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def dump_object(self, object, output_dir, filename_prefix):
        """
Dumps the object as a json to a file
        :param object:
        """
        filename = os.path.join(output_dir,
                                "{}_Objectdump_{}_{}.json".format(filename_prefix,
                                                                  datetime.datetime.strftime(datetime.datetime.now(),
                                                                                             format="%Y%m%d_%H%M%S"),
                                                                  str(uuid.uuid4())))

        with open(filename, "w") as o:
            json.dump(object, o)
