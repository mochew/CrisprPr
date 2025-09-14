from sklearn.preprocessing import LabelEncoder


def basepair_2_id_encode(sg,off):
    """
    Encode a guide RNA (sgRNA) and its off-target sequence into a fixed-length
    numeric representation (24 positions).

    Each aligned base pair (sgRNA base + off-target base) is mapped to an integer ID
    using a predefined dictionary of 36 possible combinations.
    """
    encode_dict = {'AA': 1, 'AG':2, 'AC':3, 'AT':4, 'A_':5, 'A-':6,'GG':7, 'GA':8, 'GC':9, 'GT':10, 'G_':11, 'G-':12, 'CC':13, 'CA':14, 'CG':15, 'CT':16, 'C_':17, 'C-':18, 'TT':19, 'TA':20,
                  'TG':21, 'TC':22, 'T_':23, 'T-':24, '__':25, '_A':26, '_G':27, '_C':28, '_T':29, '_-':30, '--':31, '-A':32, '-G':33, '-C':34, '-T':35, '-_':36}
    encode = []

    for idx, (s, o) in enumerate(zip(sg, off)):
        if s == 'N':
            s = o
        elif o == 'N':
            o = s
        encode.append(encode_dict[s+o])
    if len(encode) != 24:
        encode = [31]*(24-len(encode))+encode
    return encode


def sgRNA_2_numid(sg_list):
    """
    Encode a list of sgRNA identifiers (strings) into integer IDs.
    """
    encoder = LabelEncoder()
    sg_list_encoded = encoder.fit_transform(sg_list)

    sg_dict = {idx: label for idx, label in enumerate(encoder.classes_)}

    return sg_list_encoded, sg_dict