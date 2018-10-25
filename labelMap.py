label_dict = {
  "bb": 1,
  "bl": 2,
  "bc": 3,
  "cc": 4,
  "ll": 5,
  "cl": 6,
  "b2": 7,
  "H_bb": 1000,
  "H_bl": 2000,
  "H_bc": 3000,
  "H_cc": 4000,
  "H_ll": 5000,
  "H_cl": 6000,
  "top": 10,
  "dijet": 11
}

def get_single_label(nB,nC):
    label = ""
    if nB >= 1: label = "b"
    else:
        if nC >=1: label = "c"
        else: label = "l"
    return label

def get_double_label(nB1,nC1,nB2,nC2):
    label = get_single_label(nB1,nC1)+get_single_label(nB2,nC2)
    # take care of permutations
    # @TODO: return sorted(label) ?
    if label == "cb": label = "bc"
    if label == "lb": label = "bl"
    if label == "lc": label = "cl"
    return label
