#!/usr/bin/env python3

#Big O
#https://www.youtube.com/watch?v=p65AHm9MX80


def sum1(n):
    '''
    Take an input of n and return the sum of the numbers from 0 to n
    '''
    final_sum = 0
    for x in range(n+1):
        final_sum +=x
    return final_sum

def sum2(n):
    '''
    Take an input of n and return the sum of the numbers from 0 to n
    '''
    return (n*(n+1))/2

#sum1 is O(n) runtime, sum2 is O(1) runtime

"""
O(1) - is getting one specific index value, doesnt change with size 
O(n) - one for loop that increases as the loop size increases
O(n^2) - nested for loops - n operations on n items in a list = long runtime

Worst case vs. best case -- often look at worst case only  (--range--)
Time complexity vs. space complexity (memory)
"""


#----------------------------------------------------------------------
#Array Sequences
#Python unicode represented as 16bits(2 bytes)

#Dynamic Arrays - Anagram example

def anagram_builtin(s1, s2):
    #Remove spaces and lowercase letters
    s1 = s1.replace(" ","").lower()
    s2 = s2.replace(" ","").lower()
    #Remove boolean for sorted match
    return sorted(s1) == sorted(s2)


def anagram(s1, s2):
    #Remove spaces and lowercase letters
    s1 = s1.replace(" ","").lower()
    s2 = s2.replace(" ","").lower()

    #Check same number of letters 
    if len(s1) != len(s2):
        return False
    
    #count freq of each letter
    count = {}

    for letter in s1:
        if letter in count:
            count[letter] +=1
        else:
            count[letter] = 1
    #Do reverse for second string
    for letter in s2:
        if letter in count:
            count[letter] -=1
        else:
            count[letter] = 1

    for k in count:
        if count[k] != 0:
            return False
    return True
x = anagram('Clint Eastwood', 'old west action')
print(x)        #true

#----------------------------------------------------------------------
#Array Algorithms
"""
#Array Pair Sum
Given an integer array, output all the unique pairs that sum up to a specific value k.
ex. input: pair_sum([1,3,2,2], 4)  
would reutrn 2 pairs: (1,3), (2,2)
"""

def pair_sum(array, k):
    if len(array) < 2:
        return print("Too small")
    
    seen = set()
    output = set()

    for num in array:
        target = k - num

        if target not in seen:
            seen.add(num)
        else:
            output.add((min(num, target), max(num,target)))

    print('\n'.join(map(str, list(output))))

pair_sum([1,3,2,2], 4)      #(1,3) (2,2)


#----------------------------------------------------------------------
#Largest Sum
"""
Take an array with positive and negative integers
and find the maximum sum of that array
"""
def largest(arr):
    if len (arr) == 0:
        return print('Too small')
    
    max_sum = current_sum = arr[0]

    for num in arr[1:]:
        current_sum = max(current_sum + num, num)
        max_sum = max(current_sum, max_sum)
    
    return max_sum

print(largest([7,1,2,-1,3,4,10,-12,3,21,-19]))      #38


#----------------------------------------------------------------------
#Reverse a String
"""
Given a string of words, reverse all the words
ex start = "This is the best"
finish = "best the is This"
"""
def reverse_builtin(s):
    return " ".join(reversed(s.split()))

def reverse_builtin2(s):
    return s.split()[::-1]


def reverse(s):
    length = len(s)
    spaces = [' ']
    words = []
    i = 0
    while i < length:
        if s[i] not in spaces:
            word_start = i

            while i < length and s[i] not in spaces:
                i += 1
            
            words.append(s[word_start:i])

        i += 1
    
    return " ".join(reversed(words))

print(reverse("This is the best"))      #best the is This


#----------------------------------------------------------------------
#Array Analysis
"""
Given two arrays (assume no duplicates)
is 1 array a rotation of another - return True/False
same size and elements but start index is different

Big O(n) we are going through each array 2x but O(2n) = O(n) since infinite sized lists, constant mean nada

Select an indexed position in list1 and get its value. Find same element 
in list2 and check index for index from there.
If any variation then we know its false.
Getting to last item without a false means true.
"""

def rotation(list1, list2):
    if len(list1) != len(list2):
        return False
    
    key = list1[0]
    key_index = 0

    for i in range(len(list2)):
        if list2[i] == key:
            key_index = i
            
            break
    
    if key_index == 0:
        return False
    
    #modulo to check values at different indexs in two lists
    for x in range(len(list1)):
        l2index = (key_index + x) % len(list1)

        if list1[x] != list2[l2index]:
            return False
    return True

print(rotation([1,2,3,4,5,6,7], [4,5,6,7,1,2,3]))       #True


#----------------------------------------------------------------------
#Array Common Elements 
"""
Common Elements in two sorted arrays
return the common elements (as an array) between two sorted arrays of integers (Ascending order)
Example: The common elements between [1,3,4,6,7,9] and [1,2,4,5,9,10] are [1,4,9]
"""

def common_elements(a,b):
    p1 = 0
    p2 = 0
    
    result = []

    while p1 < len(a) and p2 < len(b):
        if a[p1] == b[p2]:
            result.append(a[p1])
            p1 += 1
            p2 += 1
        elif a[p1] > b[p2]:
            p2 += 1
        else:
            p1 += 1
    return result

print(common_elements([1,3,4,6,7,9], [1,2,4,5,9,10]))   #[1,4,9]


#----------------------------------------------------------------------
#Minesweeper 
"""
Write a function that will take 3 arguments:
bombs(list of bomb locations); rows; columns
minesweeper([[0,0], [0,1]], 3, 4)
    bomb at row index 0 column index 0
    bomb at row index 0 column index 1
    3 rows, 4 columns
Return an 3 x 4 array. (-1) = bomb
[[-1, -1, 1, 0],
 [ 2, 2, 1, 0],     the 2 bombs means 2 bombs in surrounding cells
  [0, 0, 0, 0]]
"""

def minesweeper(bombs, num_rows, num_cols):
    #Make array of zeros
    field = [[0 for i in range(num_cols)] for j in range(num_rows)]

    #Make bomb location -1
    for bomb_location in bombs:
        (bomb_row, bomb_col) = bomb_location
        field[bomb_row][bomb_col] = -1

        #Range around bombs
        row_range = range(bomb_row - 1, bomb_row + 2)
        col_range = range(bomb_col -1, bomb_col + 2)

        #Add +1 if tile around bomb range & not a bomb
        for i in row_range:
            current_i = i
            for j in col_range:
                current_j = j
                if (0 <= i < num_rows and 0 <= j < num_cols and field[i][j] != -1):
                    field[i][j] += 1
    return field


print(minesweeper([[0,0], [1,2]], 3, 4))        #[[-1, 2, 1, 1], [1, 2, -1, 1], [0, 1, 1, 1]]


#----------------------------------------------------------------------
#Frequent Count 
"""
Given an array what is the most frequently occuring element
O(n) linear time
"""

def most_frequent(list):
    count = {}
    max_count = 0
    max_item = None

    for i in list:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1

        if count[i] > max_count:
            max_count = count[i]
            max_item = i
    
    return max_item

print(most_frequent([1,3,3,3,2,1,1,1]))         #1


#----------------------------------------------------------------------
#Unique Characters in Strings
"""
Given a string, are all characters unique?
Should return a True or False 
Uses python builtin structures
"""

def unique_builtin(string):
    string = string.replace(" ", "")
    return len(set(string)) == len(string)     #set: unorderd set of unique elements

#print(unique_builtin('a b cdef'))                   

def unique(s):
    s = s.replace(' ', '')
    characters = set()

    for letter in s:
        if letter in characters:
            return False
        else:
            characters.add(letter)
    return True

print(unique('a b cdef'))                   #True




#----------------------------------------------------------------------
#Non Repeat Elements in Array
"""
Non repeat element
Take a string and return character that never repeats.
If multiple uniques, then return only the first unique
"""

def non_repeating(s):
    s = s.replace(' ', '').lower()
    char_count = {}

    for c in s:
        if c in char_count:
            char_count[c] += 1
        else:
            char_count[c] = 1
    """
    #To return the first unique letter only
    for c in s:
        if char_count[c] == 1:
            return c
    return None
    """
    all_uniques = []
    y = sorted(char_count.items(), key=lambda x: x[1])
    
    for item in y:
        if item[1] == y[0][1]:
            all_uniques.append(item)
    
    return all_uniques

print(non_repeating("I Apple Ape Peels"))       #[(i, 1), (s,1)]
