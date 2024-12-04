from .attacker import Attacker
ATTACKERS = {
    "base": Attacker,
}




def load_attacker(config) -> Attacker:
    return ATTACKERS[config["name"].lower()](**config)
