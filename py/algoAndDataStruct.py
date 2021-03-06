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
    
    def iter_order(self):                               # O(n??2)
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




#R10
#Depth First Search
#BFS discovers vertices reachable from queried vertex s level-by-level outward from s
#Depth First Search (DFS) also finds all vertices reachable from s, but 
# does so by searching undiscovered vertices as deep as possible before exploring other branches
#DFS searches as far as possible from first neighbor of s before searching any other neighbor of s
# Like BFS, DFS returns set of parent pointers for vertices reachable from s in 
# the order the search discovered them - forming a DFS tree
#Unlike BFS tree, a DFS tree will not represent shortest paths in an unweighted graph

def dfs(Adj, s, parent = None, order = None):   # Adj: adjacency list, s: start
    if parent is None:                      # O(1) initialize parent list
        parent = [None for v in Adj]        # O(V) (use hash if unlabeled)
        parent[s] = s                       # O(1) root
        order = []                          # O(1) initialize order array
    for v in Adj[s]:                        # O(Adj[s]) loop over neighbors
        if parent[v] is None:               # O(1) parent not yet assigned
            parent[v] = s                   # O(1) assign parent
            dfs(Adj, v, parent, order)      # Recursive call
    order.append(s)                         # O(1) amortized
    return parent, order

# recursive DFS for a graph represented using index labeled adjacency lists
#DFS recursive call is performed only when a vertex does not have a parent pointer immediately before the call
#DFS called on each vertex at most once; work done by each recursive search
# from vertex v is proportional to the out-degree deg(v) of v.
#Depth-First Search runs in O(|V | + |E|) time.


#Full Graph Exploration
#Not all vertices are reachable from a query vertex s. To explore each 
# connected component in the graph by performing a search from each vertex 
# in the graph that has not yet been discovered by the search.
#Like adding auxilary vertex with outgoing edge to every vertex in the graph then running DFS/BFS from added vertex

def full_dfs(Adj):                          # Adj: adjacency list
    parent = [None for v in Adj]            # O(V) (use hash if unlabeled)
    order = []                              # O(1) initialize order list
    for v in range(len(Adj)):               # O(V) loop over vertices
        if parent[v] is None:               # O(1) parent not yet assigned
            parent[v] = v                   # O(1) assign self as parent (a root)
            dfs(Adj, v, parent, order)      # DFS from v (BFS can also be used)
    return parent, order

#DFS is often used to refer to both a method to search a graph from a specific vertex,
# and as a method to search an entire(graph_explore)

#DFS Edge Classification
#useful to classify edges of graph in relation to dfs tree.
#Ex. graph edge from vertex u to v.
#Call edge tree edge if edge is part of the DFS tree (parent[v] = u)
#Other edges: back edge, forward edge, crosas edge. Respectively:
# For u as a descendant of v, v is descendent of u, or neither are for each other
    #track set of anscestors per vertex in DFS (as direct access array or hash)
    # if v is an ancestor of s, it certifies a back edge, cycle in graph

#Topological Sort
#A directed graph containing no directed cycle is called 
# a directed acyclic graph or a DAG
#Topological sort of a DAG is a linear ordering of the vertices such that for 
# each edge (u, v) in E, vertex u appears before vertex v in the ordering
#In the dfs function, vertices are added to the order list in the order in which their recursive DFS call finishes

#If graph is acyclic, order returned by DFS is the reverse of topo sort order
#u before v, v will start and end before dfs(u) completes = v before u in order
#v before u, u not called until v completes, else no path between(cyclic) = v added before u
#Reversing the order returned by DFS will then repre- sent a topological sort order on the vertices.

#Ex. 
'''
Run DFS on the graph(exploring the whole graph as in graph explore) 
to obtain an order of DFS vertex finishing times in O(|V | + |E|) time. 
While performing the DFS, keep track of the ancestors of each vertex in 
the DFS tree, and evaluate if each new edge processed is a back edge. 
If a back edge is found from vertex u to v, follow parent pointers back 
to v from u to obtain a directed cycle in the graph to prove to the 
principal that no such order exists. Otherwise, if no cycle is found, 
the graph is acyclic and the order returned by DFS is the reverse of a 
topological sort, which may then be returned to the principal.
'''

#https://codepen.io/mit6006/pen/dgeKEN




#R11
#Weighted Graphs
#Useful in many apps to apply numerical weight to edges in graph
#Weighted graph is then a graph G(V, E) together with weight function w: E->R (map edges to real value weights)
# weights often stored in adjacency matrix or edge obj list/set
def w(u, v):    return W[u][v]

W1 = {0: {1: -2},
      1: {2:  0},
      2: {0:  1},
      3: {4:  3}}

W2 ={0: {1: 1, 3: 2, 4: -1},    #0
    1: {0: 1},                  #1 
    2: {3: 0},                  #2 
    3: {0: 2, 2: 0},            #3 
    4: {0: -1}}                 #4

#assume that a weight function w can be stored using O(|E|) space, 
# and can return the weight of an edge in constant time
#for edge e=(u,v), use notation w(u,v) interchangeably with w(e) to refer to the weight of an edge.


#Weighted Shortest Paths
#Weighted path is path in weighted graph where the weight of the path is 
# the sum of the weights from edges in the path

#Single Source weighted shortest paths problems ask for a lowest weight path 
# to every vertex v in a graph from an input source vertex s (or indication no lowest weight path exists from s to v)

#If all edges are positive and equal to each other: simply run BFS from 
# s to min the number of edges traversed(thus minimizing the path weight)
#When edges have different/non positive weights, cant apply BFS directly

#Cycle (a path starting and ending at the same vertex) 
#Graph with cycle w/ negative weight, shortest path may not exist (run cycle to lower weight)
#Negative cyclic graph = say the shortest path from s to v is undefined, with weight ??????
#If no path exists s to v is undefined, with weight +???

#Weighted Single Source Shortest Path Algorithms
#BFS(unweighted), DAG Relaxation(DAG graph), 
# Bellman-Ford(|V|*|E| time), Dijkstra(non-negative graph, log time)


#Relaxation
#relaxation algorithm searches for a solution to an optimization problem by 
# starting with a solution that is not optimal, then iteratively improves 
# the solution until it becomes an optimal solution to the original problem

#Find the weight of shortest path from source s to each vertex v in graph
# ??(s, v)       delta 
#Init upper bound estimate d(v) on shortest path weight = +infinity
# except d(s,s) = 0  source case
# Repeatedly relax path estimate d(s,v); decrrease toward 
# true shortest path weight ??(s, v) 
#When d(s, v) = ??(s, v), we say that estimate d(s, v) is fully relaxed
#If all path estimates relaxed, solved problem

def general_relax(Adj, w, s):           # Adj: adjacency list, w: weights, s: start
    d = [float('inf') for _ in Adj]     # shortest path estimates d(s, v)
    parent = [None for _ in Adj]        # initialize parent pointers
    d[s], parent[s] = 0, s              # initialize source
    while some_edge_relaxable(Adj, w, d):       # repeat forever!
        (u, v) = get_relaxable_edge(Adj, w, d)  # relax a shortest path estimate d(s, v)
        try_to_relax(Adj, w, d, parent, u, v)
    return d, parent                    # return weights, paths via parents

#How do we ???relax??? vertices and when do we stop relaxing(end algorithm)?
#Relax an incoming edge to v from another vertex u to relax d(s,v) estimate
#Maintain d(s,u) estimate upper bounds shortest path

#Triangle Inequality
#the true shortest path weight ??(s, v) can???t be larger than d(s, u) + w(u, v)
# else edge (u,v) would be shorter path

#any time d(s, u) + w(u, v) < d(s, v), we can relax the edge by setting
#  d(s, v) = d(s, u) + w(u, v), strictly improving our shortest path estimate.

def try_to_relax(Adj, w, d, parent, u, v):
    if d[v] > d[u] + w(u, v):           # better path through vertex u
        d[v] = d[u] + w(u, v)           # relax edge with shorter path found
        parent[v] = u

#Relaxing an edge maintains d(s, v) ??? ??(s, v) for all v ??? V .
    #[estimates will never become smaller than true shortest paths]

#If no edge can be relaxed, then d(s, v) ??? ??(s, v) for all v ??? V .
    #[estimates w/ no edge can be relaxed, estimates == shortest path distances]


#Exponential Relaxation
#How many modifying edge relaxations could occur in an acyclic graph before 
# all edges are fully relaxed?
#If done in bad order, could perform exponential number of modifying relaxations
#Want to avoid for polynomial modifying edge relaxations

#DAG Relaxation
#In a directed acyclic graph (DAG), there can be no negative weight cycles, 
# so eventually relaxation must terminate.
#Relax outgoing edge from every vertex once in topological sort order of 
# the vertices correctly computes shortest paths

def DAG_Relaxation(Adj, w, s):          # Adj: adjacency list, w: weights, s: start
    _, order = dfs(Adj, s)              # run depth-first search on graph
    order.reverse()                     # reverse returned order
    d = [float('inf') for _ in Adj]     # shortest path estimates d(s, v)
    parent = [None for _ in Adj]        # initialize parent pointers
    d[s], parent[s] = 0, s              # initialize source
    for u in order:                     # loop through vertices in topo sort
        for v in Adj[u]:                # loop through out-going edges of u
            try_to_relax(Adj, w, d, parent, u, v)   # try to relax edge from u to v
    return d, parent                    # return weights, paths via parents

#DAG Relaxation algorithm computes shortest paths in a directed acyclic graph.
#topological sort order ensures that edges of the path are relaxed in the order in which they appear in the path
#Since depth- first search runs in linear time and the loops relax each edge exactly once, 
# this algorithm takes O(|V | + |E|) time.





#R12
#Bellman-Ford
#General graph that allows cycles and negative weights
#Lecture presents mod of Bellman-Ford, based on graph duplication and DAG Relaxation
# that solves SSSP in O(|V|*|E|) time and space
#Can return a negative-weight cycle reachable on a path from s to v,
#  for any vertex v with ??(s, v) = ??????.

#If ??(s, v) is finite, there exists a shortest path to v that is simple 
#Since simple paths cannot repeat vertices, finite shortest paths contain at most |V | ??? 1 edges

#Negative Cycle Witness
#k-Edge Distance ??k(s, v): the minimum weight of any path from s to v using ??? k edges

#Original Bellman-Ford detects negative weight cycle, but will not return cycle -inf

#Init distance estimates, then relax every edge in the graph in |V|-1 rounds
#If the graph does not contain negative-weight cycles, d(s, v) = ??(s, v) for all v ??? V at termination; 
# otherwise if any edge still relaxable (i.e., still violates the triangle inequality), the graph contains a negative weight cycle

def bellman_ford(Adj, w, s):                    # Adj: adjacency list, w: weights, s: start
    #Initialization
    infinity = float('inf')                     # number greater than sum of all + weights
    d =  [infinity for _ in Adj]                # shortest path estimates d(s, v)
    parent = [None for _ in Adj]                # initialize parent pointers
    d[s], parent[s] = 0, s                      # initialize source
    # construct shortest paths in rounds
    V = len(Adj)                                # number of vertices
    for k in range(V-1):                        #relax all edges in (V-1) rounds
        for u in range(V):                      # loop over all edges (u, v)    
            for v in Adj[u]:                    # relax edge from u to v
                try_to_relax(Adj, w, d, parent, u, v)
    # check for negative weight cycles accessible from s
    for u in range(V):                          # Loop over all edges (u, v)
        for v in Adj[u]:
            if d[v] > d[u] + w(u,v):            # If edge relax-able, report cycle
                raise Exception('Ack! There is a negative weight cycle!')
    return d, parent

#relates to relax paradigm, but limits order in which edges can be processed
#the algorithm relaxes every edge of the graph in a series of |V | ??? 1 rounds

#At the end of relaxation round i of Bellman-Ford, d(s, v) = ??(s, v) for any vertex v that
#has a shortest path from s to v which traverses at most i edges.

'''
If the graph does not contain negative weight cycles, some shortest path is simple, 
and contains at most |V| - 1 edges as it traverses any vertex of the graph at most once. 
Thus after |V| - 1 rounds of Bellman-Ford, d(s, v) = ??(s, v) for every vertex with a simple 
shortest path from s to v. However, if after |V| - 1 rounds of relaxation, 
some edge (u, v) still violates the triangle inequality, then there exists a path from s to v 
using |V| edges which has lower weight than all paths using fewer edges. 
Such a path cannot be simple, so it must contain a negative weight cycle.
'''

#This algorithm runs |V | rounds, where each round performs a constant amount of work for each 
# edge in the graph, so Bellman-Ford runs in O(|V ||E|) time.

#Note that this algorithm is different than the one presented in lecture in two important ways:
#1) original Bellman-Ford only keeps track of one ???layer??? of d(s,v) estimates in each round,
#  while the lecture version keeps track of dk(s, v) for k ??? {0, . . . , |V|}, which can be then used to construct negative-weight cycles.
#2)distance estimate d(s, v) in round k of original Bellman-Ford does not necessarily equal 
# dk(s, v), the k-edge distance to v computed in the lecture version
# distance estimate d(s, v) in round k of original Bellman-Ford is never larger than dk(s, v), 
# but it may be much smaller and converge to a solution quicker than the lecture version, so may be faster in practice.


#Ex. Three houses in city, find max fun route for meeting location.
#Run Bellman-Ford three times from source at each house. sum d(a,v) + d(b,v) + d(c,v) 
#find vertex that mins this sum == intersection w/ max fun for each house



#R13
#Dijkstra's Algorithm
#Most commononly used shortest path algorithm, asymptotically faster than Bellman-Ford
#Only applies to non-negative edge weight graphs
#discretizes this continuous process by repeatedly relaxing edges from a vertex whose minimum 
# weight path estimate is smallest among vertices whose out-going edges have not yet been relaxed

#Often presented in terms of a minimum priority queue data structure
#run time depends on efficiency of priority queue operations

def dijkstra(Adj, w, s):
    d = [float('inf') for _ in Adj]         # shortest path estimates d(s, v)
    parent = [None for _ in Adj]            # initialize parent pointers
    d[s], parent[s] = 0, s                  # initialize source
    Q = PriorityQueue()                     # initialize empty priority queue
    V = len(Adj)                            # number of vertices
    for v in range(V):                      # loop through vertices
        Q.insert(v, d[v])                   # insert vertex-estimate pair
    for _ in range(V):                      # main loop
        u = Q.extract_min()                 # extract vertex with min estimate
        for v in Adj[u]:                    # loop through out-going edges
            try_to_relax(Adj, w, d, parent, u, v)
            Q.decrease_key(v, d[v])         # update key of vertex
    return d, parent

# Init shortest path weight estimate and parent pointers. Init priority queue with all vertices from graph
# Main loop that extracts vertex with min estimate. Relax out-going edges from u
# Relaxation may reduce the shortest path weight estimate d(s, v), vertex v???s key in the queue must be updated (if it still exists in the queue)

#key observation is that shortest path weight estimate of vertex u equals its actual shortest path
#  weight d(s, u) = ??(s, u) when u is removed from the priority queue
#Via upper bound property,d(s, u) = ??(s, u) will still hold at termination of the algorithm


#Priority Queues
#Priority queue maintains a set of key-value pairs, where vertex v is a value and d(s, v) is its key
#Empty Init, supports 3 operations: insert(val, key), extract_min(), decrease_key(val, new_key)
#Runtime of dijksra depends on runtime of these operations:
    # TDijkstra =O(|V|??Ti +|V|??Te +|E|??Td)

#Simplest implementation is to store all the vertices and their current shortest path estimate in a dictionary
#Hash table of size O(|V|) can support expected constant time O(1) insertion and decrease-key operations,
#  though to find and extract the vertex with minimum key takes linear time O(|V|)
#If vertices and indices into the vertex set with a linear range, then can alternatively use a
# direct access array. Leads to worst case O(1) time insertion and decrease key, remains 
# linear O|V|) to find and extract the vertex with minimum key

#Runtime for Dijkstra simplifies to:   TDict = O(|V |2 + |E|)

#for dense graphs, |E| = ??(|V |2),  implementation is linear in the size of input

class PriorityQueue:                        # Hash Table Implementation
    def __init__(self):                     # stores keys with unique labels
        self.A = ()

    def insert(self, label, key):           # insert labeled key
        self.A[label] = key
    
    def extract_min(self):                  # return a label with minimum key
        min_label = None
        for label in self.A:
            if (min_label is None) or (self.A[label] < self.A[min_label].key):
                min_label = label
        del self.A[min_label]
        return min_label

    def decrease_key(self, label, key):     # decrease key of a given label
        if (label in self.A) and (key < self.A[label]):
            self.A[label] = key

#If the graph is sparse, |E| = O(|V |), we can speed things up with more sophisticated priority queue implementations.
# vertex can maintain a pointer to its stored location within the heap, or the heap can maintain 
# a mapping from values (vertices) to locations within the heap
#solution can support finding a given value in the heap in constant time
#after decreasing the value???s key, one can restore the min heap property in logarithmic time by re-heapifying the tree
#binary heap can support each of the three operations in O(log |V |) time
# the running time of Dijkstra will be:  THeap =O((|V|+|E|)log|V|)
#For sparse graphs, that???s O(|V|log |V|)
#Graphs in-between sparse and dense, Fibonacci Heap -- TFibHeap =O(|V|log|V|+|E|) run time

class Item:
    def __init__(self, label, key):
        self.label, self.key = label, key

class PriorityQueue:                        # Binary Heap Implementation
    def __init__(self):                     # stores keys with unique labels
        self.A = []
        self.label2idx = {}
    
    def min_heapify_up(self, c):
        if c == 0: return 
        p = (c-1) // 2
        if self.A[p].key > self.A[c].key:
            self.A[c], self.A[p] = self.A[p], self.A[c]
            self.label2idx[self.A[c].label] = c
            self.label2idx[self.A[p].label] = p
            self.min_heapify_up(p)
    
    def min_heapify_down(self, p): 
        if p >= len(self.A): return
        l = 2 * p + 1
        r = 2 * p + 2
        if l >= len(self.A): l = p
        if r >= len(self.A): r = p
        c = l if self.A[r].key > self.A[l].key else r
        if self.A[p].key > self.A[c].key:
            self.A[c], self.A[p] = self.A[p], self.A[c]
            self.label2idx[self.A[c].label] = c
            self.label2idx[self.A[p].label] = p
            self.min_heapify_down(c)
        
    def insert(self, label, key):           # insert labeled key
        self.A.append(Item(label, key))
        idx = len(self.A) - 1
        self.label2idx[self.A[idx].label] = idx
        self.min_heapify_up(idx)
    
    def extract_min(self):                  #remove a label with minimum key
        self.A[0], self.A[-1] = self.A[-1], self.A[0]
        self.label2idx[self.A[0].label] = 0
        del self.label2idx[self.A[-1].label]
        min_label = self.A.pop().label
        self.min_heapify_down(0)
        return min_label

    def decrease_key(self, label, key):     # decrease key of a given label
        if label in self.label2idx:
            idx = self.label2idx[label]
            if key < self.A[idx].key:
                self.A[idx].key = key
                self.min_heapify_up(idx)

#Fibonacci Heaps are not actually used very often in practice as it is more complex to implement,
#  and results in larger constant factor overhead than the other two implementations 

#When the number of edges in the graph is known to be at most linear (e.g., planar or bounded degree graphs)
#  or at least quadratic (e.g. complete graphs) in the number of vertices, then using a binary heap or dictionary 
# respectively will perform as well asymptotically as a Fibonacci Heap.

#https://codepen.io/mit6006/pen/BqgXWM




#R14
#All Pairs Shortest Paths (APSP)

#Single Source Shortest Paths (SSSP) Review
#Define/construct graphm then run SSSP algorithm on that graph - generally want to use the
# fastest SSSP algorithm that solves your problem.
#Bellman-Ford applies to any weighted graph but is the slowest, prefer others when applicable

#BFS            -- Runtime |V| + |E|  -- Restrictions: Unweighted, general graph
#DAG Relaxation -- Runtime |V| + |E|  -- Restrictions: DAG graph, ANY weights
#Dijkstra       -- Runtime |V| log |V| + |E|    -- Restrictions: Non-negative, general graph
#Bellman-Ford   -- Runtime |V| * |E|  -- Restrictions: ANY weights, general graph, slower

#Can also use algos to count connected componenets (with Full DFS or Full BFS), 
# topologically sort vertices in a DAG (using DFS), and detect negative weight cycles w/ Bellman-Ford

#APSP
#Given a weighted graph G = (V, E, w), the (weighted) APSP problem asks for the minimum 
# weight ??(u, v) of any path from u to v for every pair of vertices u, v ??? V .
#For any negative weight cycle in G, not required to return any output
#Straight forward approach = reduce to solve SSSP problem |V| times, once for each vertex in V
#sparse graph (i.e. |E| = O(|V |))
#BFS on unweighted and sparse ??(V 2) time
#Bellman-Ford on general graphs w/ negative weight edges. Runtime O(|V |2|E|), a factor of |E| larger than the output.
#Dijkstra on non-negative weight graphs, takes O(|V |2 log |V | + |V ||E|) time
#On a sparse graph, running Dijkstra |V | times is only a log |V | factor larger than the output,
#  while |V | times Bellman-Ford is a linear |V| factor larger

#Is it possible to solve the APSP problem on general weighted graphs faster than O(|V|2|E|)?


#Johnson's Algorithm
#Idea is to reduce ASPS problem on a graph with arbitrary edge weights to the ASPS problem 
# on a graph with non-negative edge weights
#Re-weighting edges in original graph to non-negative values - shortest paths remain from original graph
#Then finding shortest paths in the re-weighted graph using |V| times Dijkstra will solve the original problem.
#Johnson's idea is to assign each vertex v a real number h(v), and change the weight of each edge
# (a, b) from w(a, b) to w0(a, b) = w(a, b) + h(a) ??? h(b), to form a new weight graph G0 = (V, E, w0)
    #A shortest path (v1,v2,...,vk) in G0 is also a shortest path in G from v1 to vk.
#since each path from v1 to vk is increased by the same number h(v1) ??? h(vk), shortest paths remain shortest.

#find a vertex assignment function h, for which all edge weights w0(a, b) in the modi- fied graph are non-negative
# h = add a new node x to G with a directed edge from x to v for each vertex v ??? V to 
# construct graph G???, letting h(v) = ??(x, v). This assignment of h ensures that
#  w0(a, b) ??? 0 for every edge (a, b).
    #If h(v) = ??(x, v) and h(v) is finite, then w0(a, b) = w(a, b) + h(a) ??? h(b) ??? 0 for every edge (a, b) ??? E.
#minimum weight of any path from x to b in G??? is not greater than the minimum weight of 
# any path from x to a than traversing the edge from a to b (triangle inequality)

#Johnson???s algorithm computes h(v) = ??(x, v), negative minimum weight distances from the added node x, using Bellman-Ford
#Any -infinity deltas means negative weight cycle and can terminate w/ no output
#Otherwise, re-weighed edges are positive, can run Dijkstra |V| times on G' to find 
# a single source shortest paths distances ??0(u, v) from each vertex u in G'
# can compute each ??(u, v) by setting it to ??0(u, v)?????(x, u)+??(x, y)

#Johnson???s takes O(|V||E|) time to run Bellman-Ford, and O(|V|(|V| log |V| + |E|)) time to 
# run Dijkstra |V| times, so this algorithm runs in O(|V|2 log |V| + |V ||E|) time, 
# asymptotically better than O(|V|2|E|).




#R15
#Dynamic Programming
#Generalizes divide and conquer type recurrences when subprolem dependencies form a 
# directed acyclic graph instead of a tree.
#Often applies to optimization problems:max or min single scalar value, counting problems to count possibilities

#How to Solve a Problem Recursively (SRT BOT) "sort bot"
#1) Subproblem - subproblem x ??? X
    #Describe the meaning of a subproblem in words, in terms of parameters
    #Often subsets of input: prefixes[:i], suffixes[i:], contiguous subsequences[i:j]
    #Often record partial state: add subproblems by incrementing some auxiliary variables
#2) Relate - subproblem solutions recursively, x(i) = f(x(j),...) for one or more j < i
#3) Topological Order - argue relation is acyclic and subproblems form a DAG
#4) Base cases
    #State solutions for all (reachable) independent subproblems where relation doesn???t apply/work
#5) Original Problem
    #Show how to compute solution to original problem from solutions to subproblems
    #Possibly use parent pointers to recover actual solution, not just objective function
#6) Time analysis
    # Product[x???X] work(x), or if work(x) = O(W) for all x ??? X, then |X| ?? O(W)
    #work(x) measures nonrecursive work in relation; treat recursions as taking O(1) time

#Implementation
#after subproblems are chosen and a DAG of dependencies is found, two primary methods to solve:
#Top down - evals recursion starting from roots (vertices incident to no incoming edges)
    #End of each recursive call, solution to subproblem recorded to memo, while at the start
    # of each recursive call, the memo is checked to see if that subproblem has already been solved
#Bottom up - calculates each subproblem according to a topological sort order of the DAG of subproblem dependencies
    #Also records subproblem solutions to memo, used to solve later subproblems
    #usually subproblems are constructed so that a topological sort order is obvious, 
    #especially when subproblems only depend on subproblems having smaller parameters, so performing a DFS to find this ordering is usually unnecessary.

#the two methods are functionally equivalent but are implemented differently

#Top down is a recursive view, while Bottom up unrolls the recursion.
#Memoization is used in both implementations to remember computation from previous subproblems
#Typically memoize all evaluated subproblems, can use fewer in rounds
#Often don't just want value that is optimized, would like to return path of subproblems that resulted in optimzed value
#To reconstruct answer, maintain auxilary information in addition to the value we are optimizing
#Can maintain parent pointers to the subproblems upon which a solution to the current subproblem depends - analogous to shortest path problem pointers


#Exercise: Simplified Blackjack
#One player and dealer. Deck of cards in ordered sequence of n cards D = (c1,..., cn)
# each card ci is an integer between 1-10 inclusive (aces always 1)
#Played in rounds, dealer gets c1, c2. playr gets c3,c4 - player may draw or not
#Win = <= 21 and exceed's dealer hand, otherwise loss
#Game ends when deck is less than 5 cards remaining in the deck
#Given a deck of n cards with a known order, describe an O(n)-time algorithm to determine the maximum number of rounds the player can win by playing simplified blackjack with the deck.
#1) Subproblems -- Choose suffixes. 
    # x(i) : maximum rounds player can win by playing blackjack using cards (ci, . . . , cn)
#2) Relate -- Guess whether the player hits or not. 
    # Dealer???s hand always has value ci + ci+1
    # Player???s hand will have value either: ci+2 + ci+3 (no hit, 4 cards used in round), or ci+2 + ci+3 + ci+4 (hit, 5 cards used in round)
    # Let w(d, p) be the round result given hand values d and p (dealer and player)
        #player win:    w(d, p) = 1 if d < p ??? 21
        #player loss:   w(d, p) = 0 otherwise (if p ??? d or 21 < p)
    # x(i) = max{w(ci+ci+1,ci+2+ci+3)+x(i+4),w(ci+ci+1,ci+2+ci+3+ci+4)+x(i+5)}
    #(for n ??? (i ??? 1) ??? 5, i.e., i ??? n ??? 4)
#3) Topological Order -- Subproblems x(i) only depend on strictly larger i, so acyclic
#4) Base Case -- x(n???3)=x(n???2)=x(n???1)=x(n)=x(n+1)=0 (not enough cards for another round)
#5) Original Problem -- Solve x(i) for i ??? {1, . . . , n + 1}, via recursive top down or iterative bottom up
    #x(1): the maximum rounds player can win by playing blackjack with the full deck
#6) Time -- # subproblems: n + 1. Work per subproblem: O(1)
    # O(n) running time


#Exercise Text Justification
#Problem of fitting a sequence of n space seperated words into a column of line with constant 
# width s, to minimize the amount of white-space between words
#Words represented by width wi < s. To min white space, min badness of a line
#Assumption that line contains words from wi to wj, badness of line defined as:
    #b(i,j)=(s???(wi +...+wj))3 ifs>(wi +...+wj),andb(i,j)=???otherwise
#Want to partition words into lines to min the sum total of badness over all lines
#Microsoft word uses greedy algorithm, Latex formats to min this measure of white space w/ dynamic program

#Describe an O(n2) algorithm to fit n words into a column of width s that minimizes the sum of badness over all lines.
#1) Subproblems -- choose suffixes
    # x(i): minimum badness sum of formatting the words from wi to w(n???1)
#2) Relate -- first line must break at some owrd, try all possibilities
    # x(i) = min{b(i,j) + x(j+1)|i ??? j < n}
#3) Topological Sort -- subproblems x(i) only depend on strictly larger i, so acyclic
#4) x(n) = 0  Badness of justfying zero words is zero
#5) Solve subproblems via recursive top down or iterative bottom up 
    # solution to original problem is x(0). Store parent pointers to reconstruct line breaks
#6) Time -- of subproblems: O(n).   Work per subproblem: O(n^2)     O(n^3) runtime
    #Can we do better than O(n^3)?
#Optimizations
# badness b(i,j) could be linear -- pre-compute and remember each b(i,j) in O(1) time 
    # work per subproblem O(n) -->  O(n^2) runtime
    # Precompute also using dynamic programming
    # x(i, j): sum of word lengths wi to wj
    # x(i, j) = x(i, j ??? 1) + wj takes O(1) time to compute, faster!
    #Compute each b(i, j) = (s ??? x(i, j))3 in O(1) time

#Lecture review: SRTBOT paradigm for recursive alg. design & (with memoization) for DP algorithm design 
#Subproblem definition:  for sequence S, try 1) prefixes S[:i]  2)suffixes S[:i]    3)substrings S[i:j]
#Prefix and suffix = linear runtime O(n).  Substring is quadratic O(n^2)
#Relate subproblem solutions recursively. ID question about subproblem solution that if you knew answer reduces to smaller subproblem
#Recursive Code: def f(subprob):
#                    if subprob in memo table:              #start with empty memo = {}
#                           return memo[subprob]
#   else:            {base case check
#       memo[subprob]{recurse via relation





#R16
#Dynamic Programming Exercises
#Max Subarray Sum 
#Given array A or n integers, what is largest sum of any nonempty subarray?
#Example: A = [-9, 1, -5, 4, 3, -6, 7, 8, -2], largest subsum is 16.    (A[3:8])
#Brute forse is O(n^3), compute sum of O(n^2) subarrays in O(n) time
#faster by noticing subarray with max sum must end somewhere -- end at particular locaation k 
# in O(n) time by scanning left to k, track rolling sum and remember max
#n end locations, so alg runs in O(n^2) time -- but can go faster by reusing work from earlier scans -DP!
#1) Subproblems  -- x(k): the max subarray sum ending at A[k]
    #Prefix subproblem, but with condition Longest Increase Subsequence (LIS)
#2) Relate -- Maximizing subarray ending at k either uses item k ??? 1 or it doesn???t
    #if not, subarray is just A[k]. otherwise k-1 is used, incluse max subarray ending at k-1
    # x(k) = max{A[k], A[k] + x(k ??? 1)}
#3) Topo Order -- subproblems x(k) depend on smaller k, so acyclic
#4) Base -- x(0) = A[0] --non empty
#5) Original -- Solution to original is max of all subproblems, i.e., max{x(k) | k ??? {0, . . . , n ??? 1}}
    # Subproblems are used twice: when computing the next larger, and in the final max
#6) Time --  # of subproblems O(n),  work per subproblem O(1)   , time to solve original subproblem O(n)
    # O(n) time in total

#bottom up implementation
def max_subarray_sum(A):
    x = [None for _ in A]                       #memo
    x[0] = A[0]                                 #base case
    for k in range(1, len(A)):                  #iteration
        x[k] = max(A[k], A[k] + x[k - 1])       #relation
    return max(x)                               #original


#Edit Distance
#Plagarism detector needs to detect similarity between two texts, string A and sting B
#Measure of similarity is called the edit distance, min # of edits that will transform A -> B
# 3 operations: delete a char of A, replace char of A with another, insert between two char of A
#Describe a O(|A||B|) time algorithm to compute the edit distance between A and B.
#1) Subproblems -- modify A until last char matches B
    # x(i, j): minimum number of edits to transform prefix up to A(i) to prefix up to B(j)
#2) Relate  -- if A(i) = B(j) then match!
    #Otherwise edit to make last element of A equal to B(j) - delete, insert, or replace (Guess!)
    #Delete removes A(i). Insert add B(j) to end of A, then removes it and B(j). Replace changes A(i) to B(j) and removes both A(i)/b(j)
    #x(i,j) =   {x(i-1, j-1)                                if A(i) = B(i)
    #           {1 + min(x(i???1,j), x(i,j???1), x(i???1,j???1))    otherwise
#3) Topological order    -- Subproblems x(i, j) only depend on strictly smaller i and j, so acyclic
#4) Base Case -- x(i, 0) = i, x(0, j) = j (need many insertions or deletions)
#5) Original -- Solution to original problem is x(|A|, |B|), can store parent pointers to reconstruct edits transforming A to B
#6) Time -- # subproblems: O(n2). work per subproblem: O(1). O(n2) running time

def edit_distance(A, B):
    x = [[None] * len(A) for _ in range(len(B))]        #memo
    x[0][0] = 0                                         #base cases
    for i in range(1, len(A)):
        x[i][0] = x[i - 1][0] + 1                       # delete A[i]
    for j in range(1, len(B)):
        x[0][j] = x[0][j - 1] + 1                       # insert B[j] into A
    for i in range(1, len(A)):                          # dynamic program
        for j in range(1, len(B)):
            if A[i] == B[j]:
                x[i][j] = x[i - 1][j - 1]               # matched! no edit needed
            else:                                       # edit needed!
                ed_del = 1 + x[i - 1][j]                # delete A[i]
                ed_ins = 1 + x[i][j - 1]                # insert B[j] after A[i]
                ed_rep = 1 + x[i - 1][j - 1]            # replace A[i] with B[j]
                x[i][j] = min(ed_del, ed_ins, ed_rep)
    return x[len(A) - 1][len(B) - 1]


#Lecture examples - Longest Common Subsequence(LCS), Longest Increasing Subsequence (LIS), Alternating Coin Game 
# deal with multiple sequences, substring subproblems, parent pointers, subproblem constrint and expansion




#R17
#Dynamic Programming (cont.)
#Treasureship! example:
#Battleship alt. Place 2 x 1 ships within a 2 x n rectangular grid. Can place horizontal or vertical
#Take 2 grid squares, each has +/- int value, representing treasure to be acquired or lost at square
#Can place unlimited ships, the score of a placement of ships being the value sum of all grid squares covered by ships
#Design a placement of ships that will max your total score

#1) Subproblems - board has n columns of height 2
    # Let v(x,y) denote the grid value at row y column x , for y ??? {1,2} and x ??? {1,...,n}
    #Guess how to cover the right most squares in optimal placement - not cover, place ship vert, or ship horz
    #Exists optimal placement where no two ships aligned horizontally on top of each other
    #Lets(i,j)represent game board subset containing columns 1 to i of row 1,and columns 1 to i + j of row 2, for j ??? {0, +1, ???1}
    # x(i, j): maximum score, only placing ships on board subset s(i, j)  for i ??? {0,...,n}, j ??? {0,+1,???1}
#2) Relate 
    # If j = +1, can cover right-most square with horizontal ship or leave empty
    # If j = ???1, can cover right-most square with horizontal ship or leave empty
    # If j = 0, can cover column i with vertical ship or not cover one of right-most squares
    '''
                    max{v(i,1)+v(i-1,1)+x(i-2,+1),x(i-1,0)}             if j = -1
    x(i,j)=         max{v(i+1,2)+v(i,2)+x(i,-1),x(i,0)}                 if j = +1
                    max{v(i,1)+v(i,2)+x(i-1,0),x(i,-1),x(i-1,+1)}       if j = 0
    '''
#3) Topo - Subproblems x(i, j) only depend on strictly smaller 2i + j, so acyclic
#4) Base - s(i, j) contains 2i + j grid squares.  x(i, j) = 0 if 2i + j < 2 (can???t place a ship if fewer than 2 squares!)
#5) Original - Solution is x(n, 0), the maximum considering all grid squares. 
    #Store parent pointers to reconstruct ship locations
#6) Time - # subproblems: O(n). work per subproblem O(1).  O(n) running time


#Wafer Power
#Circuit design for highly-parallel computing
#Evenly- spaced along the perimeter of a circular wafer sits n ports for either a power source or a computing unit.
#computing unit needs energy from a power source, transferred between ports via a wire etched into the top surface of the wafer.
#But, if computing unit connected to power source too close, can overload and destroy circuit
#Also, no two etched wires may cross each other
#Given an arrangement of power sources and computing units plugged into the n ports
#describe an O(n3)-time dynamic programming algorithm to match computing units to power sources
#  by etching non-crossing wires between them onto the surface of the wafer, in order to maximize 
# the number of powered computing units, where wires may not connect two adjacent ports along the perimeter.

#1) Subproblems
    #Let (a1, . . . , an) be the ports cyclically ordered counter-clockwise around the wafer, where ports a1 and an are adjacent
    #Let ai be True if the port is a computing unit, and False if it is a power source
    # Want to match opposite ports connected by non-crossing wires
    # If match across the wafer,need to independently match ports on either side(substrings!)
    # x(i, j): maximum number of matchings, restricting to ports ak for all k ??? {i, . . . , j}
        #for i???{1,...,n}, j???{i???1,...,n}
    #j ??? i + 1 is number of ports in substring (allow j = i ??? 1 as an empty substring)
#2) Relate
    # Guess what port to match with first port in substring. Options:
        # first port does not match with anything, try to match the rest;
        # first port matches a port in the middle, try to match each side independently.
    # Non-adjacency condition restricts possible matchings between i and some port t (neighbors):
        # if (i, j) = (1, n), can???t match i with last port n or 2, so try t ??? {3, . . . , n ??? 1}
        # otherwise, just can???t match i with i + 1, so try t ??? {i + 2, . . . , j}
    # Let m(i, j) = 1 if ai =/= aj and m(i, j) = 0 otherwise (ports of opposite type match)
    #x(1,n) = max{x(2,n)} ??? {m(1,t) + x(2,t???1) + x(t+1,n)| t???{3,...,n???1}}
    #x(i,j) = max{x(i+1,j)} ??? {m(i,t) + x(i+1,t???1) + x(t+1,j) | t ??? {i+2,...,j}}
#3) Topo - Subproblems x(i, j) only depend on strictly smaller j ??? i, so acyclic
#4) Base - x(i, j) = 0 for any j ??? i + 1 ??? {0, 1, 2} (no match within 0, 1, or 2 adjacent ports)
#5) Original - Solution to original problem is x(1, n). Store parent pointers to reconstruct matching(e.g.,choice of t or no match at each step)
#6) Time  - # subproblems: O(n2). work per subproblem: O(n).    O(n^3) running time





#R18
#Dyanamic Programming [cont.]
#Subset Sum Variants
    # Input: Set of n positive integers A[i]
    # Output: Is there subset A' ??? A such that SUM(a???A' a) = S? 
    # Can solve with DP in O(nS) time 
#Subset Sum
#1) Subproblems
    # x(i, j): True if can make sum j using items 1 to i, False otherwise
#2) Relate -- Is last item i in a valid subset? (Guess!)
    #If yes, then try to sum to j ??? A[i] ??? 0 using remaining items
    #If no, then try to sum to j using remaining items
    #x(i,j)=OR    ( x(i???1,j???A[i]) ifj???A[i] ) 
    #               x(i ??? 1, j) always     )     for i ??? {0,...,n},j ??? {0,...,S}
#3) Topological Order - Subproblems x(i, j) only depend on strictly smaller i, acyclic
#4) Base -  x(i,0) = True for i ??? {0,...,n} (trivial to make zero sum!)
    #       x(0, j) = False for j ??? {1, . . . , S} (impossible to make positive sum from empty set)
#5) Original - Maximum evaluated expression is given by x(n, S)
#6) Time - (# subproblems: O(nS)) ?? (work per subproblem O(1)) = O(nS) running time.

#Knapsack example: 
#Input: Knapsack with size S, want to fill with items each item i has size si and value vi.
#Output: A subset of items (may take 0 or 1 of each) with P si ??? S maximizing value P vi
#(Subset sum same as 0-1 Knapsack when each vi = si, deciding if total value S achievable)
#Solution: Subset with max value is all items except the last one (greedy fails)
    #If yes, get value vi and pack remaining space S ??? si using remaining items
    #If no, then try to sum to S using remaining items

#1) Subproblem - x(i, j): maximum value by packing knapsack of size j using items 1 to i
#2) Relate
    #       x(i,j)= max (vi+x(i???1,j???si) ifj???s ) 
    #                   x(i ??? 1, j) always                
#3) Topo - depend on small i
#4) Base - x(i,0) = 0 for i ??? {0,...,n} (zero value possible if no more space)
    # x(0,j) = 0 for j ??? {1,...,S} (zero value possible if no more items)
#5) Original -- Maximum evaluated expression is given by x(n, S), store parent pointers to get items
#6) Time    # subproblems: O(nS)    work per subproblem O(1)    O(nS) running time

#Subset Sum         https://codepen.io/mit6006/pen/JeBvKe
#Knapsack           https://codepen.io/mit6006/pen/VVEPod




#R19
#Complexity
#Knapsack Problem Revisited
    #Input: Knapsack with volume S, want to fill with items: item i has size si and value vi.
    #Output: A subset of items (may take 0 or 1 of each) with SUM (si ??? S) maximizing SUM (vi)
    #Solvable in O(nS) time via dynamic programming
#How does run time compare to input?
    #Input size? if numbers written in binary, input has size O(n log S) bits
    #O(nS) runs in exponential time compared to input
    #If numbers polynomially bounded, S = nO(1), then dynamic program is polynomial
#Called pseudopolynomial time algorithm
#Can it be solved in polynomial time when numbers are not polynomially bounded?
# No if P =/= NP        (polynomial time doesnt equal non-deterministic polynomial time)

#Decision Problems
#Decision problem: assignment of inputs to No (0) or Yes (1)
#Inputs are either No instances or Yes instances (i.e. satisfying instances)
#ex. shortest path(s-t), Negative cycle, Longest Path, Subset Sum, Tetris, Chess, Halting Problems
#Algorithm/Program: constant length code (working on a word-RAM with ??(log n)-bit words) to solve
#  a problem, i.e., it produces correct output for every input and the length of the code is 
# independent of the instance size
#Decidable if program exists to solve problem in finite time(not infinity)

#Decidability
#A program is finite string of bits (program is a file made of strings made of bits)
#(# of programs |N|, countably infinite) << (# of problems |R|, uncountably infinite)
    #Proves that most decision problems not solvable by any program (undecidable)
    #Fortunately most problems we think of are algorithmic in structure and are decidable
#Decidable Problem Classes
    #R = finite time (recursive)    EXP = exponential time 2nO(1) (most problems here)
    #P = polynomial time nO(1) (efficient algorithms)
    #sets are distinct, i.e. P < EXP < R

#Nondeterministic Polynomial Time (NP)
    #P is the set of decision problems for which there is an algorithm A such that for every 
    # instance I of size n, A on I runs in poly(n) time and solves I correctly
    #NP is the set of decision problems for which there is an algorithm V , a ???verifier???, that 
    # takes as input an instance I of the problem, and a ???certificate??? bit string of length 
    # polynomial in the size of I, so that:
        #V always runs in time polynomial in the size of I
        #if I is a YES-instance, then there is some certificate c so that V on input (I , c) returns YES
        #if I is a NO-instance, then no matter what c is given to V together with I, V will always output NO on (I , c)
    #You can think of the certificate as a proof that I is a YES-instance. If I is actually a NO- instance then no proof should work
    #P within NP (if you can solve the problem, the solution is a certificate)
    #Most people think Most people think P ( NP (( EXP), i.e.,t generating solutions harder than checking
        #If can show a problem is hardest problem in NP, then problem cannot be solved in polynomial time if P /= NP

#Reductions
#Want to solve problem A. Can convert A into problem B you know how to solve
    #Solve using algorithm for B and use it to compute solution to A
#Called a Reduction from problem A to problem B (A ??? B)
#Because B can be used to solve A, B is at least as hard (A ??? B)

#General algorithmic strategy: reduce to a problem you know how to solve
    #ex. A(unweighted shortest paths) -> convert (give equal lengths 1) -> B(weighted shortest path)
#Problem A is NP-Hard if every problem in NP is polynomially reducible to A
    # A is at least as hard as(can be used to solve) every problem in NP (X ??? A for X ??? NP)

#NP-Complete = NP ??? NP-Hard

#All NP-Complete problems are equivalent, i.e. reducible to each other
#First NP-Complete? Every decision problem reducible to satisfying a logical circuit
    #ex. longest path, tetris are np-complete, chess is exp-complete

#               NP-Hard->       EXP-Hard->
# < --- | --- NP-Complete --- EXP-complete --- | --->
#     <-P       <-NP            <-EXP         <-R

#Knapsack problem = NP-hard. Reduce: Partition
# Input: List of n numbers ai   Output: Does there exist a partition into two sets with equal sum?
#Reduction: si = vi = ai, S = 1  SUM(ai)
#Knapsack at least as hard as Partition, so since Partition is NP-Hard, so is Knapsack 
# Knapsack in NP, so also NP-Complete




#R20
#Course Review
#Goals - 1) Solve hard computational problems (non constant-sized inputs)
        #2) Argue algorithm is correct (induction, recursion)
        #3) Argue an algorithm is "good" (asymptotics, model of computation)
        #4) Effectively communicate all three aboe points

#Do there always exist good algorithms? No!
    #Most not solvable efficiently. Polynomial = polynomial size of input. Pseudopolynomial = poly in size of input AND size of numbers in input
    #NP = nondeterministic polynomial time, are certificates polynomially checkable
        #NP-hard: set of problems that can be used to solve any problem in NP poly time
        #NP-complete: intersection of NP-hard and NP  |

#How do you solve algorithms problem?
#Reduce to a problem you know how to solve
    #Search/Sort
        #Search: Extrinsic(Sequence) and Intrinsic (Set) data structures
        #Sort: Comparison Model, Stability, In-place
    #Graphs
        #Reachability, Connected Components, Cycle Detection, Topological Sort
        #Single-Source / All-Pairs Shortest Paths

#Design of a novel Recursive Algorithm
    #Brute Force
    #Divide and Conquer
    #Dynamic Programming -- SRT BOT
    #Greedy/Incremental

#further OCW 6.046 Design & Analysis of Algorithms
    #grad: 6.851: Advanced Data Structures, 6.854: Advanced Algorithms
    #Cryptography (6.875), Biology (6.047), Game Theory (6.853), Graphics (6.837), Vision (6.819), Geometry (6.850)

#6.046 = extension of 6.006
#Data Structures: Union-find, Amortization via potential analysis
#Graphs: Min Spanning Trees, Network Flows/Cuts
#Algorithm Design (Paradigms): Divide and Conquer, Dynammic Programming, Greedy
#Complexity: Reductions

#Relax Problems
    #Randomized Algorithms
        # Las Vegas: always correct, probably fast (like hashing)
        # Monte Carlo: always fast, probably correct
        #generally get faster randomized algorithms on structured data
    #Numerical Algorithms/Continuous Optimization
        #Approximate real numbers! Pay time for precision
    #Approximation Algorithms
        #Input optimization problem (min/max over weighted outputs)
        #Many optimization problems NP-hard
        #Can we get close to optimal solution in polynomial time?

#Change Model of Computation
    #In 6.006 used the Word-RAM model - bits representation
    #Cache Models (memory hierarchy cost model)
    #Quantum Computer (exploiting quantum properties)
    #Parallel Processors (use multiple CPUs instead of just one)
        #Multicore, large shared memory
        #Distributed cores, message passing


#End of recitation material for 6.006 
