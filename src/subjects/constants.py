DIP_IMU_LOCATIONS = [
    "head",
    "sternum",
    "pelvis",
    "lshoulder",
    "rshoulder",
    "lupperarm",
    "rupperarm",
    "llowerarm",
    "rlowerarm",
    "lupperleg",
    "rupperleg",
    "llowerleg",
    "rlowerleg",
    "lhand",
    "rhand",
    "lfoot",
    "rfoot",
]

NIMBLE_BODY_NODES_DIP = [
    "head",
    "thorax",
    "pelvis",
    "scapula_l",
    "scapula_r",
    "humerus_l",
    "humerus_r",
    "ulna_l",
    "ulna_r",
    "femur_l",
    "femur_r",
    "tibia_l",
    "tibia_r",
    "hand_l",
    "hand_r",
    "calcn_l",
    "calcn_r"
]

UIP_IMU_LOCATIONS = [
    "lelbow",
    "relbow",
    "lknee",
    "rknee",
    "upperneck",
    "root"
]

NIMBLE_BODY_NODES_UIP = [
    "hand_l",
    "hand_r",
    "tibia_l",
    "tibia_r",
    "head",
    "pelvis"
]

TOTAL_CAPTURE_IMU_LOCATIONS = [
    "Pelvis",
    "L_UpLeg",
    "R_UpLeg",
    "L_LowLeg",
    "R_LowLeg",
    "L_Foot",
    "R_Foot",
    "Sternum",
    "Head",
    "L_UpArm",
    "R_UpArm",
    "L_LowArm",
    "R_LowArm"
]

NIMBLE_BODY_NODES_TOTAL_CAPTURE = [
    "pelvis",
    "femur_l",
    "femur_r",
    "tibia_l",
    "tibia_r",
    "talus_l",
    "talus_r",
    "lumbar_body",
    "head",
    "humerus_l",
    "humerus_r",
    "hand_l",
    "hand_r"
]

NIMBLE_BODY_NODES_ALL = [
    'head', 
    'thorax', 
    'pelvis', 
    'scapula_l', 
    'scapula_r', 
    'humerus_l', 
    'humerus_r', 
    'ulna_l', 
    'ulna_r', 
    'femur_l', 
    'femur_r', 
    'tibia_l', 
    'tibia_r', 
    'hand_l', 
    'hand_r', 
    'calcn_l', 
    'calcn_r'
]

_sets = [
    set(NIMBLE_BODY_NODES_DIP),
    set(NIMBLE_BODY_NODES_UIP),
    set(NIMBLE_BODY_NODES_TOTAL_CAPTURE),
]
_counts = [sum(node in s for s in _sets) for node in NIMBLE_BODY_NODES_ALL]
_invs = [(1/c if c else 0) for c in _counts]
_total = sum(_invs)
NIMBLE_BODY_NODE_WEIGHTS = [_inv / _total for _inv in _invs]
del _sets, _counts, _invs, _total
