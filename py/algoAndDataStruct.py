#https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/

#R1
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
   

#R2
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


#R3
class Sorted_Array_Set:
    def __init__(self):     self.A = Array_Seq()        #O(1)
    def __len__(self):      return len(self.A)          #O(1)
    def __iter__(self):     yield from self.A           #O(n)
    def iter_order(self):   yield from self             #O(n)

    def build(self, X):                                 #O(?)
        self.A.build(X)
        self._sort()

    def _sort(self):                                    #O(?)
        pass

    def _binary_search(self, k, i, j):                  #O(log n)
        if i >= j:  return i 
        m = (i+j) // 2
        x = self.A.get_at(m)
        if x.key > k:       return self._binary_search(k, i, m - 1)
        if x.key < k:       return self._binary_search(k, m + 1, j)
        return m

    def find_min(self):                                 #O(1)
        if len(self) > 0:   return self.A.get_at(0)
        else:               return None
    
    def find_max(self):                                 #O(1)
        if len(self) > 0:   return self.A.get_at(len(self) - 1)
        else:               return None
    
    def find(self, k):                                  #O(log n)
        if len(self) == 0:  return None
        i = self._binary_search(k, 0, len(self) - 1)
        x = self.A.get_at(i)
        if x.key == k:      return x
        else:               return None

    def find_next(self, k):                             #O(log n)
        if len(self) == 0:  return None
        i = self._binary_search(k, 0, len(self) - 1)
        x = self.A.get_at(i)
        if x.key > k:       return x
        if i + 1 < len(self): return self.A.get_at(i + 1)
        else:               return None

    def find_prev(self, k):                             #O(log n)
        if len(self) == 0:  return None
        i = self._binary_search(k, 0, len(self) - 1)
        x = self.A.get_at(i)
        if x.key < k:       return x
        if i>0:             return self.A.get_at(i - 1)
        else:               return None

    def insert(self, x):                                #O(n)
        if len(self.A) == 0:
            self.A.insert_first(x)
        else:
            i = self._binary_search(x.key, 0, len(self.A) - 1)
            k = self.A.get_at(i).key
            if k == x.key:
                self.A.set_at(i, x)
                return False
            if k > x.key:   self.A.insert_at(i, x)
            else:           self.A.insert_at(i + 1, x)
        return True

    def delete(self, k):                                #O(n)
        i = self._binary_search(k, 0, len(self.A) - 1)
        assert self.A.get_at(i).key == k
        return self.A.delete_at(i)


#Selection Sort
#maintains and grows a subset the largest i items in sorted order
def selection_sort(A):                      # Selection sort array A
    for i in range(len(A) - 1, 0, -1):      # O(n) loop over array
        m = i                               # O(1) initial index of max
        for j in range(i):                  # O(i) search for max in A[:i]
            if A[m] < A[j]:                 # O(1) check for larger value
                m = j                       # O(1) new max found
        A[m], A[i] = A[i], A[m]             # O(1) swap


#Insertion Sort
#maintains and grows a subset of the first i input items in sorted order
def insertion_sort(A):                      #Insertion sort array A
    for i in range(1, len(A)):              #O(n) loop over array
        j = i                               #O(1) initialize pointer
        while j > 0 and A[j] < A[j-1]:      #O(i) loop over prefix
            A[j-1], A[j] = A[j], A[j-1]     #O(1) swap
            j = j - 1                       #O(1) decrement j


#Both - In-place - means can be implemented using at most a constant amount of additional space
#Insertion = stable - means items having the same value will appear in the sort in the same order as they appeared in the input array


#Merge Sort - Theta(n log n)
#recursively sorts the left and right half of the array, and then merges the two halves in linear time
def merge_sort(A, a = 0, b = None):         # Sort sub-array A[a:b]
    if b is None:                           # O(1) initialize
        b = len(A)                          # O(1)
    if 1 < b - a:                           # O(1) size k = b - a
        c = (a + b + 1) // 2                # O(1) compute center
        merge_sort(A, a, c)                 # T(k/2) recursively sort left
        merge_sort(A, c, b)                 # T(k/2) recursively sort right
        L, R = A[a:c], A[c:b]               # O(k) copy
        i, j = 0, 0                         # O(1) initialize pointers
        while a < b:                        # O(n)
            if (j >= len(R)) or (i < len(L) and L[i] < R[j]):   # O(1) check side
                A[a] = L[i]                 # O(1) merge from left
                i=i+1                       # O(1) decrement left pointer
            else:
                A[a] = R[j]                 # O(1) merge from right
                j = j + 1                   # O(1) decrement right pointer
            a= a+1                          # O(1) decrement merge pointer

#merge sort not in place
#https://codepen.io/mit6006/pen/RYJdOG      #https://codepen.io/mit6006/pen/wEXOOq

#Sacrifice some time in building the data structure to speed up order queries - technique called preprocessing


#R4
#Comparison model - worst case = height of algorithm's decision tree (log n)
#Find(k) operation is most used in interfaces; sorted arrays and balanced BinarySearchTrees are able to support find(k) asymptotically optimally

#Direct Access Array 
# shallow branch as large as space in computer - worst case constant time search at cost of storage space, u time for order operations
#use hashing to overcome space obstacle
class DirectAccessArray:
    def __init__(self, u):  self.A = [None] * u         # O(u)
    def find(self, k):      return self.A[k]            # O(1)
    def insert(self, x):    self.A[x.key] = x           # O(1)
    def delete(self, k):    self.A[k] = None            # O(1)
    def find_next(self, k):
        for i in range(k, len(A)):                 # O(u)
            if self.A[i] is not None:
                return A[i]
    def find_max(self):
        for i in range(len(A) - 1, -1, -1):        # O(u)
            if self.A[i] is not None:
                return A[i]
    def delete_max(self):
        for i in range(len(A) - 1, -1, -1):        # O(u)
            x = A[i]
            if x is not None:
                A[i] = None
                return x

from locale import ABDAY_1
import random
from re import A
#Hash collisions if more points than space, pigeonhole principle. 
#Store collisions somewhere else in same direct accesss array or elsewhere. Open addressing(in practice) or chaining
#Chain as collision resolution strategy - allow find, insert, delete
class Hash_Table_Set:
    def __init__(self, r = 200):                        # O(1)
        self.chain_set = Set_from_Seq(Linked_List_Seq)
        self.A = []
        self.size = 0
        self.r = r                                      # 100/self.r = fill ratio
        self.p = 2**31 - 1
        self.a = random.randint(1, self.p - 1)
        self._compute_bounds()
        self._resize(0)
    
    def __len__(self):  return self.size                # O(1)
    def __iter__(self):                                 # O(n)
        for X in self.A:
            yield from X
        
    def build(self, X):                                 # O(n)e
        for x in X: self.insert(x)
    
    def _hash(self, k, m):                              # O(1)
        return ((self.a * k) % self.p) % m

    def _compute_bounds(self):                          # O(1)
        self.upper = len(self.A)
        self.lower = len(self.A) * 100*100 // (self.r*self.r)
    
    def _resize(self, n):                               # O(n)
        if (self.lower >= n) or (n >= self.upper):
            f = self.r // 100
            if self.r % 100:     f += 1
            #f = ceil(r / 100)
            m= max(n, 1) * f
            A= [self.chain_set() for _ in range(m)]
            for x in self:
                h = self._hash(x.key, m)
                A[h].insert(x)
            self.A = A
            self._compute_bounds()

    def find(self, k):                                  # O(1)e
        h = self._hash(k, len(self.A))
        return self.A[h].find(k)
    
    def insert(self, x):                                # O(1)ae
        self._resize(self.size + 1)
        h = self._hash(x.key, len(self.A))
        added = self.A[h].insert(x)
        if added:   self.size += 1
        return added
    
    def delete(self, k):                                # O(1)ae
        assert len(self) > 0
        h = self._hash(k, len(self.A))
        x = self.A[h].delete(k)
        self.size -= 1
        self._resize(self.size)
        return x
    
    def find_min(self):                                 # O(n)
        out = None
        for x in self:
            if (out is None) or (x.key < out.key):
                out = x
        return out

    def find_max(self):                                 # O(n)
        out = None
        for x in self:
            if (out is None) or (x.key > out.key):
                out = x
        return out

    def find_next(self, k):                             # O(n)
        out = None
        for x in self:
            if x.key > k:
                if (out is None) or (x.key < out.key):
                    out = x
        return out
    
    def find_prev(self, k):                             # O(n)
        out = None
        for x in self:
            if x.key < k:
                if (out is None) or (x.key > out.key):
                    out = x
        return out
    
    def iter_order(self):                               # O(nˆ2)
        x = self.find_min()
        while x:
            yield x
            x = self.find_next(x.key)



#R5
#Comparison Sorting - omega(n log n)
#Direct Access Array Sort - no duplicate keys, and can't handle large key ranges
def direct_access_sort(A):
    """Sort A assuming items have distinct non-negative keys"""
    u = 1 + max([x.key for x in A])                 # O(n) find maximum key
    D = [None] * u                                  # O(u) direct access array
    for x in A:                                     # O(n) insert items
        D[x.key] = x
    i = 0
    for key in range(u):                            ## O(u) read out items in order
        if D[key] is not None:
            A[i] = D[key]
            i += 1


#Counting Sort
#link chain to each direst access array index (allows dupllicate keys)
#stable(items appear in the same order in the output as the input) - seq. queue interface
def counting_sort(A):
    """Sort A assuming items have non-negative keys"""
    u = 1 + max([x.key for x in A])                 # O(n) find maximum key
    D = [[] for i in range(u)]                      # O(u) direct access array of chains
    for x in A:                                     # O(n) insert into chain at x.key
        D[x.key].append(x)
    i = 0
    for chain in D:                                 # O(u) read out items in order
        for x in chain:
            A[i] = x
            i += 1
    
#counting sort alt implementation -> compute final index location of each item via cumulative sums
def counting_sort(A):
    """Sort A assuming items have non-negative keys"""
    u= 1 + max([x.key for x in A])                  # O(n) find maximum key
    D = [0] * u                                     # O(u) direct access array
    for x in A:                                     # O(n) count keys
        D[x.key] += 1
    for k in range(1, u):                           # O(u) cumulative sums
        D[k] += D[k - 1]
    for x in list(reversed(A)):                     # O(n) move items into place
        A[D[x.key] - 1] = x
        D[x.key] -= 1



#Tuple Sort  - break int keys into parts, sort each part
#uses a stable sorting algorithm as a subroutine to repeatedly sort the objects, first according to the least important key, then the second least important key, all the way up to most important key
#similar to how one might sort on multiple rows of a spreadsheet by different columns
#Need stalbe algo - only correct if previous rounds of sorting are maintained


#Radix Sort
#Increase rangeof int sets you can sort in linear time, break into multiples of powers of n
#representing each item key its sequence of digits when represented in base n
#sort digit representations with tuple sort by sorting on each digit in order from least significant to most significant digit using counting sort.
def radix_sort(A):
    """Sort A assuming items have non-negative keys"""
    n = len(A)
    u = 1 + max([x.key for x in A])                 # O(n) find maximum key
    c = 1 + (u.bit_length() // n.bit_length())
    class Obj: pass
    D = [Obj() for a in A]
    for i in range(n):                              # O(nc) make digit tuples
        D[i].digits = []
        D[i].item = A[i]
        high = A[i].key
        for j in range(c):                          # O(c) make digit tuple
            high, low = divmod(high, n)
            D[i].digits.append(low)
    for i in range(c):                              # O(nc) sort each digit
        for j in range(n):                          # O(n) assign key i to tuples
            D[j].key = D[j].digits[i]
        counting_sort(D)                            # O(n) sort on digit i
    for i in range(n):                              # O(n) output to A
        A[i] = D[i].item

#https://codepen.io/mit6006/pen/LgZgrd


