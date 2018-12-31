class PriorityQueue(): 
    def __init__(self): 
        self.q = []
        self.q.append(None) # offset = 1
  
    def __str__(self): 
        return(' '.join([str(i) for index, i in enumerate(self.q) if index != 0]))
  
    def isEmpty(self): 
        '''for checking if the q is empty'''
        return(len(self.q) == 1) 
    
    def _swap(self, i, j):
        self.q[i], self.q[j] = self.q[j], self.q[i]
 
    def insert(self, data): 
        '''for inserting an element in the q'''
        if self.isEmpty() is not True:
            self.q.append(data)
            idx = len(self.q) - 1
            parent_idx = int(idx/2)
            while self.q[idx] > self.q[parent_idx]:
                self._swap(idx, parent_idx)
                idx = parent_idx
                parent_idx = int(idx/2)
                if parent_idx == 0:
                    break
        else:
            self.q.append(data)

    def peek(self):
        '''return the first item in queue and not delete it'''
        return(self.q[1] if len(self.q) > 1 else None)

    def _maxHeapify(self, rootIdx):
        '''Make sure the subtree under root meets the max heap rule'''
        leftIdx = rootIdx * 2
        rightIdx = rootIdx * 2 + 1
        maxIdx = rootIdx
        
        if leftIdx < len(self.q):
            maxIdx = leftIdx if self.q[leftIdx] > self.q[maxIdx] else maxIdx
            if rightIdx < len(self.q):
                maxIdx = rightIdx if self.q[rightIdx] > self.q[maxIdx] else maxIdx

        if maxIdx != rootIdx:
            self._swap(rootIdx, maxIdx)
            self._maxHeapify(maxIdx)

    def delete(self, data):
        if len(self.q) == 1:
            pass
        elif len(self.q) == 2:
            idx = self.q[1:].index(data) + 1
            del self.q[idx]
        else:
            idx = self.q[1:].index(data) + 1
            self.q[idx] = self.q[-1]
            del self.q[-1]
            self._maxHeapify(idx)

