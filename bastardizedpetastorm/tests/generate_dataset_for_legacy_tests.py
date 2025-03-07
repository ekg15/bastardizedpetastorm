#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import bastardizedpetastorm
from bastardizedpetastorm.tests.test_common import create_test_dataset


def generate_dataset_for_legacy_test():
    """Generates a test dataset and stores it into petastorm/tests/data/legacy/x.x.x folder. The version number
    is acquired automatically from bastardizedpetastorm.__version__"""
    dataset_name = petastorm.__version__
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'legacy', dataset_name)
    url = 'file://' + path

    create_test_dataset(url, range(100))


if __name__ == '__main__':
    generate_dataset_for_legacy_test()
