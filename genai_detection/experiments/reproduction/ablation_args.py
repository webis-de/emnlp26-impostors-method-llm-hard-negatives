# Copyright 2026 Klara M. Gutekunst, Webis
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

from ablations import (ASGALFImpostorDetector, BDIImpostorDetector,
                       HBCImpostorDetector,
                       Potha2017ImpostorDetector,
                       PermutationCalibratedImpostorDetector,
)
from ablations.std_impostor import StdImpostor


ABLATION_DETECTORS = {
    "bdi": BDIImpostorDetector,
    "homotopy": HBCImpostorDetector,
    "potha2017": Potha2017ImpostorDetector,
    "asgalf": ASGALFImpostorDetector,
    "std_impostor": StdImpostor,
    "permutation_calibrated": PermutationCalibratedImpostorDetector,
}

ABLATION_ARGS = {
        "bdi": {
            "rounds": 100,   # Nagy (2024) does not specify beyond "repeat n times", but BDI implementation has 100 nb_bootstrap_iter
            "portion_delete": 0.67, # cf. Nagy (2024)
            "n_impostors": 30,   # Nagy (2024) does not specify, but BDI implementation has 30 nb_impostors
        },
        "homotopy":{
            "rounds": 50,
            "n_impostors": 2,
            },
        "asgalf":{
            "rounds": 50,
            "portion_delete": 0.6,
            "n_impostors": 20,
            },
        "std_impostor": {
            "rounds": 50,
            "portion_delete": 0.6,
            "n_impostors": 20,
                },
        "permutation_calibrated": {
            "rounds": 100,
            "portion_delete": 0.5,
            "n_impostors": 25,
            "n_permutations": 199,
                },
        "potha2017": {
                "rounds": 10,
                "portion_delete": 0.5,
                "impostors_per_problem": 50,
                "impostors_per_round": 5,
                }
        }
