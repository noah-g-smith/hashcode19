

def cost(photo1, photo2):
    intersect = len(photo1.intersection(photo2))
    return min(len(photo1) - intersect, len(photo2) - intersect, intersect)


def sequence_cost(sequence, photos):
    total_cost = 0
    for i in range(len(sequence) - 1):
        if sequence[i + 1] == -1:
            break

        if isinstance(sequence[i], tuple):
            old_tags = photos[sequence[i][0]][1].union(photos[sequence[i][1]][1])
        else:
            old_tags = photos[sequence[i]][1]

        if isinstance(sequence[i + 1], tuple):
            new_tags = photos[sequence[i + 1][0]][1].union(photos[sequence[i + 1][1]][1])
        else:
            new_tags = photos[sequence[i + 1]][1]

        total_cost += cost(old_tags, new_tags)
    return total_cost
