from .defender import Defender
from .cube_defender import CUBEDefender, CasualCUBEDefender
from .graceful_defender import GraCeFulDefender

DEFENDERS = {
    "base": Defender,
    'cube': CUBEDefender,
    'casualcube':CasualCUBEDefender,
    'graceful':GraCeFulDefender
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
