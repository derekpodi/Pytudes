#https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/


class StaticArray:
    def __init__(self, n):
        self.data = [None] * n
    def get_at(self, i):
        if not (0 <= i < len(self.data)): raise IndexError
        return self.data[i]

    def set_at(self, i, x):
        if not (0 <= i < len(self.data)): raise IndexError
        self.data[i] = x


def birthday_match(students):
    """
    Find a pair of students with the same birthday
    Input: tuple of student (name, bday) tuples
    Output: tuple of student names or None
    """
    n = len(students) # O(1)
    record = StaticArray(n) # O(n)
    for k in range(n): #n 
        (name1, bday1) = students[k] # O(1) 
        for i in range(k): #k
            (name2, bday2) = record.get_at(i)
            if bday1 ==bday2:
                 return (name1, name2)
        record.set_at(k, (name1, bday1))
    return None


class Array_Seq: 
    def __init__(self):      #O(1)
        self.A = [] 
        self.size = 0

    def __len__(self):      return self.size    #O(1) 
    def __iter__(self):     yield from self.A   #O(n) iter_seq

    def build (self, X):                        # O(n) 
        self.A = [a for a in X]     # pretend this builds a static array 
        self.size = len(self.A) 

    def get_at(self, i):   return self.A[i]     #O(1) 
    def set_at(self, i, x): self.A[i] = x       #O(1)

    def _copy_forward(self, i, n, A, j):        #O(n)
        for k in range(n):
            A[j + k] = self.A[i + k]
        
    def _copy_backward(self, i, n, A, j):       #O(n)
        for k in range(n - 1, -1, -1):
            A[j + k] = self.A[i + k]
    
    def insert_at(self, i, x):                  #O(n)
        n = len(self) 
        A=[None]*(n+1) 
        self._copy_forward(0, i, A, 0) 
        A[i]=x
        self._copy_forward(i, n - i, A, i + 1)
        self.build(A)

    def delete_at(self, i):                     #O(n)
        n = len(self)
        A=[None]*(n-1)
        self._copy_forward(0, i, A, 0)
        x = self.A[i]
        self._copy_forward(i + 1, n - i - 1, A, i) 
        self.build(A)
        return x

    def insert_first(self, x):  self.insert_at(0,x)             #O(n)
    def delete_first(self):      return self.delete_at(0)
    def insert_last(self, x):   self.insert_at(len(self), x)
    def delete_last(self):      return self.delete_at(len(self) - 1)



class Linked_List_Node:
    def __init__(self, x):          #O(1)
        self.item = x
        self.next = None

    def later_node(self, i):        #O(1)
        if i == 0:  return self
        assert self.next
        return self.next.later_node(i - 1)

    
    
class Linked_List_Seq:
    def __init__(self):             #O(1)
        self.head = None
        self.size = 0

    def __len__(self):  return self.size    #O(1)

    def __iter__(self):             #O(n) iter_seq
        node = self.head
        while node:
            yield node.item
            node = node.next
    
    def build(self, X):             #O(n)
        for a in reversed(X):
            self.insert_first(a)
    
    def get_at(self, i):            #O(i)
        node = self.head.later_node(i)
        return node.item
    
    def set_at(self, i, x):         #O(i)
        node = self.head.later_node(i)
        node.item = x
    
    def insert_first(self, x):      #O(1)
        new_node = Linked_List_Node(x)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def delete_first(self):         #O(1)
        x = self.head.item
        self.head = self.head.next
        self.size -= 1
        return x
    
    def insert_at(self, i, x):      #O(i)
        if i==0:
            self.insert_first(x)
            return
        new_node = Linked_List_Node(x)
        node = self.head.later_node(i - 1)
        new_node.next = node.next
        node.next = new_node
        self.size += 1

    def delete_at(self, i):         #O(i)
        if i==0:
            return self.delete_first()
        node = self.head.later_node(i - 1)
        x = node.next.item
        node.next = node.next.next
        self.size -= 1
        return x
    
    def insert_last(self, x):   self.insert_at(len(self), x)    #O(n)
    def delete_last(self):      return self.delete_at(len(self) - 1)


    
    
class Dynamic_Array_Seq(Array_Seq):
    def __init__(self, r = 2):              #O(1)
        super().__init__()
        self.size = 0
        self.r = r
        self._compute_bounds()
        self._resize(0)
    
    def __len__(self):  return self.size    #O(1)

    def __iter__(self):                     #O(n)
        for i in range(len(self)): yield self.A[i]
    
    def build(self, X):                     #O(n)
        for a in X: self.insert_last(a)

    def _compute_bounds(self):              #O(1)
        self.upper = len(self.A)
        self.lower = len(self.A) // (self.r * self.r)
    
    def _resize(self, n):                   #O(1) or O(n)
        if (self.lower < n < self.upper): return 
        m = max(n, 1) * self.r
        A=[None]*m
        self._copy_forward(0, self.size, A, 0) 
        self.A = A
        self._compute_bounds()
    
    def insert_last(self, x):               #O(1)amortorized  (Python append)
        self._resize(self.size + 1)
        self.A[self.size] = x
        self.size += 1
    
    def delete_last(self):                  #O(1)amortorized  (Python pop)
        self.A[self.size - 1] = None
        self.size -= 1
        self._resize(self.size)
    
    def insert_at(self, i, x):              #O(n)
        self.insert_last(None)
        self._copy_backward(i,self.size - (i + 1), self.A, i + 1)
        self.A[i] = x
    
    def delete_at(self, i):                 #O(n)
        x = self.A[i]
        self._copy_forward(i + 1, self.size - (i + 1), self.A, i)
        self.delete_last()
        return x
                                            #O(n)
    def insert_first(self, x):  self.insert_at(0, x)
    def delete_first(self):     return self.delete_at(0)


    

# Non efficient implementation of SEQ -> SET
def Set_from_Seq(seq):
    class set_from_seq:
        def __init__(self):     self.S = seq()
        def __len__(self):      return len(self.S)
        def __iter__(self):     yield from self.S

        def build(self, A):
            self.S.build(A)
        
        def insert(self, x):
            for i in range(len(self.S)):
                if self.S.get_at(i).key == x.key:
                    self.S.set_at(i, x)
                    return
            self.S.insert_last(x)

        def delete(self, k):
            for i in range(len(self.S)):
                if self.S.get_at(i).key == k:
                    return self.S.delete_at(i)
        
        def find(self, k):
            for x in self:
                if x.key == k:  return x
            return None

        def find_min(self):
            out = None
            for x in self:
                if (out is None) or (x.key < out.key):
                    out = x
            return out
        
        def find_max(self):
            out = None
            for x in self:
                if (out is None) or (x.key > out.key):
                    out = x
            return out

        def find_next(self, k):
            out = None
            for x in self:
                if x.key > k:
                    if (out is None) or (x.key < out.key):
                        out = x
            return out
        
        def find_prev(self, k):
            out = None
            for x in self:
                if x.key < k:
                    if (out is None) or (x.key > out.key): 
                        out =x
            return out
        
        def iter_ord(self):
            x = self.find_min()
            while x:
                yield x
                x = self.find_next(x.key)
    return set_from_seq
        
