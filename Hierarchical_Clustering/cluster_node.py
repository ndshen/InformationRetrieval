class Cluster_Node():
    def __init__(self, i:int, j:int, sim=None):
        self.i = min(i, j)
        self.j = max(i, j)
        self.sim = sim
    def __str__(self):
        return("({} {} {})".format(self.i, self.j, self.sim))
    def __lt__(self, other):
        return(self.sim < other.sim)
    def __le__(self, other):
        return(self.sim <= other.sim)
    def __gt__(self, other):
        return(self.sim > other.sim)
    def __ge__(self, other):
        return(self.sim >= other.sim)
    def __eq__(self, other):
        return(self.i == other.i and self.j == other.j)
    def __ne__(self, other):
        return(self.i != other.i or self.j != other.j)

class Cluster():
    def __init__(self, c:set, idx_name_mapping:list):
        self.c = c
        self.map = idx_name_mapping
    def __str__(self):
        return(" ".join(sorted([self.map[i] for i in self.c], key= lambda x: int(x[:-4]))))
    def __iter__(self):
        for doc_name in sorted([self.map[i] for i in self.c], key= lambda x: int(x[:-4])):
            yield doc_name
    def __lt__(self, other):
        return(min(self.c) < min(other.c))
    def __gt__(self, other):
        return(min(self.c) > min(other.c))
    def __eq__(self, other):
        return(min(self.c) == min(other.c))
    def add(self, data):
        self.c.add(data)
    def merge(self, other):
        self.c = self.c.union(other.c)