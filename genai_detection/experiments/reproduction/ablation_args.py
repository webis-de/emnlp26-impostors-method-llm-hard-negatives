from ablations import (ASGALFImpostorDetector, BDIImpostorDetector,
                       HBCImpostorDetector,
                       Potha2017ImpostorDetector,
)
from ablations.std_impostor import StdImpostor

ABLATION_DETECTORS = {
    "bdi": BDIImpostorDetector,
    "homotopy": HBCImpostorDetector,
    "potha2017": Potha2017ImpostorDetector,
    "asgalf": ASGALFImpostorDetector,
    "std_impostor": StdImpostor,
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
        "potha2017": {
                "rounds": 10,
                "portion_delete": 0.5,
                "impostors_per_problem": 50,
                "impostors_per_round": 5,
                }
        }
