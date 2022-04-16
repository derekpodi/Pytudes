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



#R6
#Binary Trees - tree graph of binary nodes
# contains pointer to an item stored at node, pointer to parent node (possible none), 
# pointer to left child node (possible none), pointer to a right child node (possible none)
class Binary_Node:
    def __init__(A, x):                             # O(1)
        A.item = x
        A.left   = None
        A.right  = None
        A.parent = None
        # A.subtree_update()

#One root node (no parent), traverse from leaf(no children) to parent via pointers
#Nodes set passed in traversal to root are called ancestors
#Depth of node <X> in subtree rooted at <R> is length of path from <X> back to <R>
#Height of node <X> is max depth of any node in the subtree rooted at <X>

#Binary Tree - no node is more than O(log n) pointer hops from root
#keep height low - operations run O(log n) rather than linear

#Traversal Order
#Natural order based on left and right children
#Every node in the left subtree of node <X> comes before <X> in the traversal order; and
#every node in the right subtree of node <X> comes after <X> in the traversal order
#recursively list nodes in left subtree, root, recursive right subtree - O(n)
def subtree_iter(A):                                # O(n)
    if A.left:   yield from A.left.subtree_iter()
    yield A
    if A.right:  yield from A.right.subtree_iter()

#Tree Navigation
def subtree_first(A):                               # O(h)
        if A.left:  return A.left.subtree_first()
        else:       return A

def subtree_last(A):                                # O(h)
        if A.right: return A.right.subtree_last()
        else:       return A

#Next node in traveral order == successor
#Previous node in traveersal order == predecessor
def successor(A):                       # O(h)
    if A.right: return A.right.subtree_first()
    while A.parent and (A is A.parent.right):
        A = A.parent
    return A.parent

def predecessor(A):                     # O(h)
    if A.left:  return A.left.subtree_last()
    while A.parent and (A is A.parent.left):
        A = A.parent
    return A.parent

#Dynamic Operations on BT
#To add or remove items in binary tree, must take care to preserve traversal order
#To add before in traversal order: add as left child, if left child spot is taken; add to right child of subtree from left child node
def subtree_insert_before(A, B):        # O(h)
    if A.left:
        A = A.left.subtree_last()
        A.right, B.parent = B, A
    else:
        A.left,  B.parent = B, A
    # A.maintain()

def subtree_insert_after(A, B):
    if A.right:
        A = A.right.subtree_first()
        A.left,  B.parent = B, A
    else:
        A.right, B.parent = B, A
    # A.maintain()

#To delete node: if leaf, simply delete. 
# If not a leaf, swap the node's item with the item in the node's successor or predecessor down the tree until the item is in a leaf which can be removed
def subtree_delete(A):                  # O(h)
    if A.left or A.right:               # A is not a leaf
        if A.left:  B = A.predecessor()
        else:       B = A.successor()
        A.item, B.item = B.item, A.item
        return B.subtree_delete()
    if A.parent:                        #A is a leaf
        if A.parent.left is A:  A.parent.left = None
        else:                   A.parent.right = None
        # A.parent.maintain()
    return A


#Binary Node Full Implementation
class Binary_Node:
    def __init__(A, x):                 # O(1)
        A.item   = x
        A.left   = None
        A.right  = None
        A.parent = None
        # A.subtree_update()
    
    def subtree_iter(A):                # O(n)
        if A.left:   yield from A.left.subtree_iter()
        yield A
        if A.right:  yield from A.right.subtree_iter()
    
    def subtree_first(A):               # O(h)
        if A.left:  return A.left.subtree_first()
        else:       return A

    def subtree_last(A):                # O(h)
        if A.right: return A.right.subtree_last()
        else:       return A

    def successor(A):                       # O(h)
        if A.right: return A.right.subtree_first()
        while A.parent and (A is A.parent.right):
            A = A.parent
        return A.parent
    
    def predecessor(A):                     # O(h)
        if A.left:  return A.left.subtree_last()
        while A.parent and (A is A.parent.left):
            A = A.parent
        return A.parent

    def subtree_insert_before(A, B):        # O(h)
        if A.left:
            A = A.left.subtree_last()
            A.right, B.parent = B, A
        else:
            A.left,  B.parent = B, A
        # A.maintain()
    
    def subtree_insert_after(A, B):         # O(h)
        if A.right:
            A = A.right.subtree_first()
            A.left, B.parent =B,A 
        else:
            A.right, B.parent =B,A
        # A.maintain()
    
    def subtree_delete(A):                  # O(h)
        if A.left or A.right:
            if A.left:  B = A.predecessor()
            else:       B = A.successor()
            A.item, B.item = B.item, A.item
            return B.subtree_delete()
        if A.parent:
            if A.parent.left is A:  A.parent.left = None
            else:                   A.parent.right = None
            # A.parent.maintain()
        return A
    

#Top Level Data Structure
#previous within binary_tree class to apply to any subtree
#Here is general binary tree DS that stores a pointer to its root, and number of items it stores
class Binary_Tree:
    def __init__(T, Node_Type = Binary_Node):
        T.root = None
        T.size = 0
        T.Node_Type = Node_Type

    def __len__(T): return T.size
    def __iter__(T):
        if T.root:
            for A in T.root.subtree_iter():
                yield A.item


#Exercise Problem
"""
#Build Tree from arrray keeping index as traverse order and height O(log n)
#Solve by setting middle as root, follow subtree traverse rules, balanced tree height for log n

def build(X):
    A = [x for x in X]
    def build_subtree(A, i, j):
        c = (i+j) // 2
        root = self.Node_Type(A[c])
        if i < c:                   # needs to store more items in left subtree
            root.left = build_subtree(A, i, c - 1)
            root.left.parent = root
        if c < j:                   # needs to store more items in right subtree
            root.right = build_subtree(A, c + 1, j)
            root.right.parent = root
        return root
    self.root = build_subtree(A, 0, len(A)-1)   
"""

#Binary Tree to implement a Set interface
#use traversal order to store the items sorted in increasing key order
#Binary Search Tree Property: left sub keys < node key < right sub keys
#can walk tree to find query key, recursing appropriate side
class BST_Node(Binary_Node):                    #Set Binary Tree == BST
    def subtree_find(A, k):                     # O(h)
        if k < A.item.key:
            if A.left:  return A.left.subtree_find(k)
        elif k > A.item.key:
            if A.right: return A.right.subtree_find(k)
        else:           return A
        return None

    def subtree_find_next(A, k):                 # O(h)
        if A.item.key <= k:
            if A.right: return A.right.subtree_find_next(k)
            else:       return None
        elif A.left:
            B = A.left.subtree_find_next(k)
            if B:       return B
        return A

    def subtree_find_prev(A, k):                # O(h)
        if A.item.key >= k: 
            if A.left:  return A.left.subtree_find_prev(k)
            else:       return None
        elif A.right:
            B = A.right.subtree_find_prev(k)
            if B:       return B
        return A
    
    def subtree_insert(A, B):                   # O(h)
        if B.item.key < A.item.key:
            if A.left:  A.left.subtree_insert(B)
            else:       A.subtree_insert_before(B)
        elif B.item.key > A.item.key:
            if A.right: A.right.subtree_insert(B)
            else:       A.subtree_insert_after(B)
        else:    A.item = B.item

class Set_Binary_Tree(Binary_Tree):         # Binary Search Tree
    def __init__(self): super().__init__(BST_Node)

    def iter_order(self): yield from self

    def build(self, X):
        for x in X: self.insert(x)

    def find_min(self):
        if self.root:   return self.root.subtree_first().item

    def find_max(self):
        if self.root:   return self.root.subtree_last().item

    def find(self, k):
        if self.root:
            node = self.root.subtree_find(k)
            if node:    return node.item
    
    def find_next(self, k):
        if self.root:
            node = self.root.subtree_find_next(k)
            if node:    return node.item
    
    def find_prev(self, k):
        if self.root:
            node = self.root.subtree_find_prev(k)
            if node:    return node.item

    def insert(self, x):
        new_node = self.Node_Type(x)
        if self.root:
            self.root.subtree_insert(new_node)
            if new_node.parent is None: return False
        else:
            self.root = new_node
        self.size += 1
        return True
    
    def delete(self, k):
        assert self.root
        node = self.root.subtree_find(k)
        assert node
        ext = node.subtree_delete()
        if ext.parent is None:  self.root = None
        self.size -= 1
        return ext.item



#R7
#Balanced Binary Tree  -- height (log n)
#AVL Tree == every node is height-balanced
#left and right subtrees differ in height by at most 1
#Skew is height of right subtree minus height of left subtree [-1, 0, 1] = balanced

#Rotations - O(1)
#change structure of tree for balance without changing traversal order
"""
_____<D>__      rotate_right(<D>)     __<B>_____
 __<B>__     <E>        =>            <A>    __<D>__
  <A>  <C>    /\                       /\     <C> <E> 
  /\    /\  /___\       <=            /__\     /\  /\ 
/___\ /___\           rotate_left(<B>)        /__\/__\ 

"""
#Can change depth of nodes and preserve traversal order
def subtree_rotate_right(D):
    assert D.left
    B, E = D.left, D.right
    A, C = B.left, B.right
    D, B = B, D
    D.item, B.item = B.item, D.item
    B.left, B.right = A, D
    D.left, D.right = C, E
    if A: A.parent = B
    if E: E.parent = D
    # B.subtree_update()
    # D.subtree_update()

def subtree_rotate_left(B): #O(1)
    assert B.right
    A, D = B.left, B.right
    C, E = D.left, D.right
    B,D = D,B
    B.item, D.item = D.item, B.item
    D.left, D.right = B, E 
    B.left, B.right = A, C 
    if A: A.parent = B
    if E: E.parent = D
    # B.subtree_update()
    # D.subtree_update()

#Maintain Height Balance
#Addition or deletion of a leaf can change height balance, affects ancestors of leaf
#Walk from leaf imbalance to the root, rebalance along the way (at most O(log n) rotations)
def skew(A):                            # O(?)
        return height(A.right) - height(A.left)

def rebalance(A):
    if A.skew() == 2:
        if A.right.skew() < 0:
            A.right.subtree_rotate_right()
        A.subtree_rotate_left()
    elif A.skew() == -2:
        if A.left.skew() > 0:
            A.left.subtree_rotate_left()
        A.subtree_rotate_right()
    
def maintain(A):                            #O(h)
    A.rebalance()
    A.subtree_update()
    if A.parent: A.parent.maintain()

def height(A):                          # Omega(n)
    if A is None: return -1
    return 1 + max(height(A.left), height(A.right))

#Rebalance takes omega(log n), to rebalance at most O(log n) time we need to eval height in O(1) time
#Speed up via augmentation: each node stores and maintains the value of its own subtree height
#eval now reading stored value in O(1) time - when tree structure changes, update to recompute height at nodes
def height(A):
    if A:   return A.height
    else:   return -1

    def subtree_update(A):                  # O(1)
        A.height = 1 + max(height(A.left), height(A.right))

#Store added info at each node to quickly query in future
#To augment nodes of binary tree, need: defined property subtree corresponds to,
#and show how to compute in O(1) time from the augmentation children


#Binary Node Implementation with AVL Balancing
def height(A):
    if A:   return A.height
    else:   return -1

class Binary_Node:
    def __init__(A, x):                     # O(1)
        A.item   = x
        A.left   = None
        A.right  = None
        A.parent = None
        A.subtree_update()

    def subtree_update(A):                  # O(1)
        A.height = 1 + max(height(A.left), height(A.right))
    
    def skew(A):                            # O(1)
        return height(A.right) - height(A.left)
    
    def subtree_iter(A):                    # O(n)
        if A.left:   yield from A.left.subtree_iter()
        yield A
        if A.right:  yield from A.right.subtree_iter()
    
    def subtree_first(A):                   # O(log n)
        if A.left:  return A.left.subtree_first()
        else:       return A

    def subtree_last(A):                    # O(log n)
        if A.right: return A.right.subtree_last()
        else:       return A

    def successor(A):                       # O(log n)
        if A.right: return A.right.subtree_first()
        while A.parent and (A is A.parent.right):
            A = A.parent
        return A.parent
    
    def predecessor(A):                     # O(log n)
        if A.left:  return A.left.subtree_last()
        while A.parent and (A is A.parent.left):
            A = A.parent
        return A.parent

    def subtree_insert_before(A, B):        # O(log n)
        if A.left:
            A = A.left.subtree_last()
            A.right, B.parent = B, A
        else:
            A.left,  B.parent = B, A
        A.maintain()

    def subtree_insert_after(A, B):         # O(log n)
        if A.right:
            A = A.right.subtree_first()
            A.left, B.parent =B,A 
        else:
            A.right, B.parent =B,A 
        A.maintain()

    def subtree_delete(A):                  # O(log n)
        if A.left or A.right:
            if A.left:  B = A.predecessor()
            else:       B = A.successor()
            A.item, B.item = B.item, A.item
            return B.subtree_delete()
        if A.parent:
            if A.parent.left is A:  A.parent.left  = None
            else:                   A.parent.right = None
            A.parent.maintain()
        return A
    
    def subtree_rotate_right(D):            # O(1)
        assert D.left
        B, E = D.left, D.right
        A, C = B.left, B.right
        D, B = B, D
        D.item, B.item = B.item, D.item
        B.left, B.right = A, D
        D.left, D.right = C, E
        if A: A.parent = B
        if E: E.parent = D
        B.subtree_update()
        D.subtree_update()
    
    def subtree_rotate_left(B):             # O(1)
        assert B.right
        A, D = B.left, B.right
        C, E = D.left, D.right
        B, D = D, B
        B.item, D.item = D.item, B.item
        D.left, D.right = B, E 
        B.left, B.right = A, C 
        if A: A.parent = B
        if E: E.parent = D
        B.subtree_update()
        D.subtree_update()

    def rebalance(A):                       # O(1)
        if A.skew() == 2:
            if A.right.skew() < 0:
                A.right.subtree_rotate_right()
            A.subtree_rotate_left()
        elif A.skew() == -2:
            if A.left.skew() > 0:
                A.left.subtree_rotate_left()
            A.subtree_rotate_right()
    
    def maintain(A):                        # O(log n)
        A.rebalance()
        A.subtree_update()
        if A.parent:    A.parent.maintain()
    

#The Binary_Node maintains balance; supports Binary_Tree_Set operations in O(log n) time, except build and iter (O(n log n), O(n) respectively)
#This forms AVL Tree == Set AVL

#To use Binary Tree to implement Sequence interface, we use traversal order of tree to store items in seq order
#Find on seq tree would be O(n) time, use stored subtree size to compare index
# and then recurse on correct side
class Size_Node(Binary_Node):
    def subtree_update(A):                  # O(1)
        super().subtree_update()
        A.size = 1
        if A.left:   A.size += A.left.size
        if A.right:  A.size += A.right.size
    
    def subtree_at(A, i):                   # O(h)
        assert 0 <= i
        if A.left:      L_size = A.left.size
        else:           L_size = 0
        if i < L_size:  return A.left.subtree_at(i)
        elif i > L_size: return A.right.subtree_at(i - L_size - 1)
        else:           return A

#can find ith node in balanced binary tree in O(log n) time
#Can build tree from input seq in O(n) time -- Seequence AVL

#https://codepen.io/mit6006/pen/NOWddZ

class Seq_Binary_Tree(Binary_Tree):
    def __init__(self): super().__init__(Size_Node)

    def build(self, X):
        def build_subtree(X, i, j):
            c = (i + j) // 2
            root = self.Node_Type(A[c])
            if i < c:
                root.left = build_subtree(X, i, c - 1)
                root.left.parent = root
            if c < j:
                root.right = build_subtree(X, c + 1, j)
                root.right.parent = root
            root.subtree_update()
            return root
        self.root = build_subtree(X, 0, len(X) - 1)
        self.size = self.root.size
    
    def get_at(self, i):
        assert self.root
        return self.root.subtree_at(i).item
    
    def set_at(self, i, x):
        assert self.root
        self.root.subtree_at(i).item = x
    
    def insert_at(self, i, x): 
        new_node = self.Node_Type(x) 
        if i==0:
            if self.root:
                node = self.root.subtree_first()
                node.subtree_insert_before(new_node)
            else:
                self.root = new_node
        else:
            node = self.root.subtree_at(i - 1)
            node.subtree_insert_after(new_node)
        self.size += 1
    
    def delete_at(self, i):
        assert self.root
        node = self.root.subtree_at(i)
        ext = node.subtree_delete()
        if ext.parent is None:  self.root = None
        self.size -= 1
        return ext.item

    def insert_first(self, x):  self.insert_at(0, x)
    def delete_first(self):     return self.delete_at(0)
    def insert_last(self, x):   self.insert_at(len(self), x)
    def delete_last(self):      return self.delete_at(len(self) - 1)


#R8
#Priority Queues
"""
algorithm_|_data structure   _|_insertion_|_extraction_|_total       
Selection | Sort Array        | O(1)      | O(n)       | O(n^2)
Insertion | Sort Sorted Array | O(n)      | O(1)       | O(n^2)
Heap Sort | Binary Heap       | O(log n)  | O(log n)   | O(n log n)
"""
#Base class - interface of priority queue, maintains internal array A of 
# items, implements insert() and delete_max() --subclasses
class PriorityQueue:
    def __int__(self):
        self.A = []

    def insert(self, x):
        self.A.append(x)
    
    def delete_max(self):
        if len(self.A) < 1:
            raise IndexError('pop from empty priority queue')
        return self.A.pop()                     #NOT correct on its own
    
    @classmethod
    def sort(Queue, A):
        pq = Queue()                            #make empty priority queue
        for x in A:                             #n x T_insert
            pq.insert(x)
        out = [pq.delete_max() for _ in A]      #n x T_delete_max
        out.reverse()
        return out

#sort two loops over array: one to insert all elements, another to populate 
# the output array with successive maxima in reverse order


#Array Heaps
#Implementations of Selection sort and Merge Sort from the perspective of priority queues

class PQ_Array(PriorityQueue):                  #Selection Sort
    # PriorityQueue.insert already correct: appends to end of self.A
    def delete_max(self):                       # O(n)
        n, A, m = len(self.A), self.A, 0
        for i in range(1, n):
            if A[m].key < A[i].key:
                m = i
        A[m], A[n] = A[n], A[m]                 # swap max with end of array
        return super().delete_max()             # pop from end of array


class PQ_SortedArray(PriorityQueue):            #Insertion Sort
    # PriorityQueue.delete_max already correct: pop from end of self.A
    def insert(self, *args):                    # O(n)
        super().insert(*args)                   # append to end of array
        i, A = len(self.A) - 1, self.A          # restore array ordering
        while 0 < i and A[i + 1].key < A[i].key:
            A[i + 1], A[i] = A[i], A[i+1]
            i -= 1

#use *args to allow insert to take one argument or zero args - zero args used when making priority queues in place


#Binary Heaps
#takes advantage of the logarithmic height of a complete binary tree to improve performance
#max_heapify_up and max_heapify_down handle bulk of work
class PQ_Heap(PriorityQueue):
    def insert(self, *args):                    # O(log n)
        super().insert(*args)                   #append to the end of array
        n, A = self.n, self.A
        max_heapify_up(A, n, n-1)
    
    def delete_max(self):                       # O(log n)
        n, A = self.n, self.A
        A[0], A[n] = A[n], A[0]
        max_heapify_down(A, n, 0)
        return super().delete_max()             #pop from end of array

#compute parent and child indices, given an index representing a node in a tree whose root is the first element of the array
def parent(i):
    p = (i-1) // 2
    return p if 0 < i else i

def left(i, n):
    l = 2 * i + 1
    return l if l < n else i

def right(i, n):
    r = 2 * i + 2
    return r if r < n else i

#Assume nodes in A[:n] satisfy Max-Heap Property except for node A[i]
#must maintain the root = largest key, move up or down to satisfy Max-Heap Property list
def max_heapify_up(A, n, c):                    # T(c) = O(log c)
    p = parent(c)                               # O(1) index of parent (or c)
    if A[p].key < A[c].key:                     # O(1) compare
        A[c], A[p] = A[p], A[c]                 # O(1) swap parent
        max_heapify_up(A, n, p)                 # T(p) = T(c/2) recursive call on parent

def max_heapify_down(A, n, p):                  # T(p) = O(log n - log p)
    l, r = left(p, n), right(p, n)              # O(1) indices of children (or p)
    c = l if A[r].key < A[l].key else r         # O(1) index of largest child
    if A[p].key < A[c].key:                     # O(1) compare
        A[c], A[p] = A[p], A[c]                 # O(1) swap child
        max_heapify_down(A, n, c)               # T(c) recursive call on child


# O(n) Build Heap
#repeated max_heap insertion takes O(n log n) time
#Build in linear time if whole array is accessible - construct in reverse level order(leafs to root)
# while maintaining that all nodes processed maintain Max-Heap Property by running max_heapify_down at each node
# last half of array are all leaves, so don't need to run max_heapify_down on them
def build_max_heap(A):
    n = len(A)
    for i in range(n // 2, -1, -1):              # O(n) loop backward over array
        max_heapify_down(A, n, i)               # O(log n - log i) fix max heap


#In-Place Heaps
#modify base class PriorityQueue to take an entire array A of elements,
# and maintain the queue itself in the prefix of the first n elements of A(where n <= len(A))
#Insert now inserts the item already stored in A[n], incorporates it into the now-larger queue
#Delete_max deposits output into A[n] before decreasing size
class PriorityQueue:
    def __init__(self, A):
        self.n, self.A = 0, A
    
    def insert(self):                           #absorb element A[n] into the queue
        if not self.n < len(self.A):
            raise IndexError('insert into full priority queue')
        self.n += 1

    def delete_max(self):                       #remove element A[n -1] from the queue
        if self.n < 1:
            raise IndexError('pop from empty priority queue')
        self.n -= 1                             #Not correct on its own!
    
    @classmethod
    def sort(Queue, A):
        pq = Queue(A)                           #make empty priority queue
        for i in range(len(A)):                 # n x T_insert
            pq.insert()
        for i in range(len(A)):                 # n x T_delete_max
            pq.delete_max()
        return pq.A
    
#PQ_Heap known as heap sort

#https://codepen.io/mit6006/pen/KxOpep



#R9
#Graphs

#A graph G = (V, E) == mathematical object comprising a set of vertices V (also called nodes)
# and a set of edges E, each edge E is a two-element subset of vertices from V

#Vertex and edge are incident or adjacent if the edge contains the vertex

#Edge is directed if the subset pair is directed (u, v) one way
#Edge is undirected if subset pair is unordered {u, v}  set - (u,v), (v, u)

#Edge follow tail ---> head, outgoing ---> incoming edge
# undirected graphs every edge is incoming and outgoing

#In degree and out degree of vertex v denotes # of incoming and outgoing edges connected to v
# degree generally refers to out-degree

#Path in a graph is a sequence of vertices such that for every ordered pair 
# of vertices(vi, vi+1) there exists an outgoing edge in the graph from vi to vi+1
#Length of path is # of edges in the path (1 less that # of vertices)

#Strongly Connected if path from every node to every other node in graph(undirected)


#Graph Representations
#Common to store the adjacencies of vertex v, the set of vertices that are accessible from v via a single outgoing edge
#Adjacency list - outgoing neighbor vertices fro each vertex
#can store within direct access array; array slot i points to adjacency list of vertex labeled i
A1 =    [[1],       #0      #Ordered example
        [2],        #1
        [0],        #2
        [4],        #3
        []]         #4 
A2 =    [[1, 4, 3], #0      #Unordered example
        [0],        #1
        [3],        #2
        [0, 2],     #3
        [0]]        #4 

#array structure useful to loop over edges incident to a vertex
# edges appear at most twice; obstacle is determining if edge in graph 
# takes omega(|V|) time. Overcome by using hash table (edge check O(1))
S1 = {0: {1},       
      1: {2},       
      2: {0},       
      3: {4}}
S2 = {0: {1, 3, 4}, #0 
      1: {0},       #1 
      2: {3},       #2 
      3: {0, 2},    #3 
      4: {0}}       #4


#Breadth-First Search
#Given a graph, a common query is to find the vertices reachable by 
# a path from a queried vertex s.
#A breadth-first search (BFS) from s discovers the level sets of s: level Li 
# is the set of vertices reachable from s via a shortest path of length i (not reachable via a path of shorter length).

#i=0, L = {s} (the vertex itself). i + 1 is any vertex reachable from from s,
# it must have an incoming edge from a vertex whose shortest path from s has length i, so it is contained in level Li.

#Parent labels (pointers) together determine a BFS tree from vertex s, 
# containing some shortest path from s to every other vertex in the graph.
def bfs(Adj, s):                            # Adj: adjacency list, s: starting vertex
    parent = [None for v in Adj]            # O(V) (use hash if unlabeled)
    parent[s] = s                           # O(1) root
    level = [[s]]                           # O(1) initialize levels
    while 0 < len(level[-1]):               # O(?) last level contains vertices
        level.append([])                    # O(1) amortized, make new level
        for u in level[-2]:                 # O(?) loop over last full level
            for v in Adj[u]:                # O(Adj[u]) loop over neighbors
                if parent[v] is None:       # O(1) parent not yet assigned
                    parent[v] = u           # O(1) assign parent from level[-2]
                    level[-1].append(v)     # O(1) amortized, add to border
    return parent

#Inner loop repeated at most O(|E|) times, outer loop cycles all deg(v) outgoing edges from vertex v
#Breadth-First Search runs in O(|V| + |E|) time


#Use parent labels returned by breadth-first search to construct shortest path from vertex s to vertex t
# follow parent pointers from t backward through the graph to s
# code to compute shortest path from s to t (worst case run time O(|V| + |E|))
def unweighted_shortest_path(Adj, s, t):
    parent = bfs(Adj, s)                    # O(V+E) BFS tree from s
    if parent[t] is None:                   # O(1) t reachable from s?
        return None                         # O(1) no path
    i = t                                   # O(1) label of current vertex
    path = [t]                              # O(1) initialize path
    while i != s:                           # O(V) walk back to s
        i = parent[i]                       # O(1) move to parent
        path.append(i)                      # O(1) amortized add to path
    return path[::-1]                       # O(V) return reversed path



