from rtree import index

class Tree(object):
    def __init__(self, X):
        """
        Tree representation
        :param X: Search Space
        """
        p = index.Property(dimension=3)
        self.V = index.Index(interleaved=True, properties=p)  # vertices in an rtree
        self.V_count = 0
        self.E = {}  # edges in form E[child] = parent
