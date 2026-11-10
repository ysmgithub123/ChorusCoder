"""Source-span overlap used by support segmentation."""

def overlaps(first, second):
    return any((max(a, c) < min(b, d) for a, b in first for c, d in second))
