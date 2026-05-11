# Adapted from partialLS/utils/optWrapper.py
def getOptNameRoot(optName, ignore_string):
    offset_idx = len(ignore_string)
    if optName[-2:] in ["-0", "-1", "-2"]:
        partial_linesearch_type = float(optName[-1])
        offset_idx += 2
    else:
        partial_linesearch_type=0

    if "layer-pos" in optName:
        per_parameter, pos_only = True, True
        optNameRoot = optName[:-(10+offset_idx)]
    elif "-pos" in optName:
        per_parameter, pos_only = False, True
        optNameRoot = optName[:-(4+offset_idx)]
    elif "-layer" in optName:
        per_parameter, pos_only = True, False
        optNameRoot = optName[:-(6+offset_idx)]
    else:
        per_parameter, pos_only = False, False
        optNameRoot = optName[:-offset_idx]

    #print(f"getOptNameRoot: optNameRoot={optNameRoot}, per_parameter={per_parameter}, pos_only={pos_only}, partial_linesearch_type={partial_linesearch_type}")

    return optNameRoot, per_parameter, pos_only, partial_linesearch_type