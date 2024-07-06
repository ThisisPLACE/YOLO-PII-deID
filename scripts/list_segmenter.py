from math import ceil


def segment(inlist, seg, part):
    nseg = int(ceil(len(inlist) / seg))
    i = part
    outlist = []
    if i == nseg - 1:
        for n in range(0, ((seg - (nseg * seg - len(inlist))))):
            c = i * seg + n
            outlist.append(inlist[c])

    else:
        for n in range(0, seg):
            c = i * seg + n
            outlist.append(inlist[c])

    return outlist
