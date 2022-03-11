#!/usr/bin/python3

#Python Built In Functions - examples and reference
#https://docs.python.org/3/library/functions.html
#https://treyhunner.com/2019/05/python-builtins-worth-learning/


###Commonly Known Built-In Functions####
#1) #print
words = ["Welcome", "to", "Python"]
print(words)
print(*words, end="!\n")
print(*words, sep="\n")

#2) #len
words = ["Welcome", "to", "Python"]
len(words)

#3) #str
#Can't concatenate strings and numbers in Python, need to manually convert
version = 3
"Python " + str(version)

#4) #int
#convert str -> ints. Truncates floats to ints
program_name = "Python 3"
version_number = program_name.split()[-1]
int(version_number)
from math import sqrt
sqrt(28)
int(sqrt(28))  #// operator more appropriate

#5) #float
#Can use to convert ints to floating point numbers
program_name = "Python 3"
version_number = program_name.split()[-1]
float(version_number)
pi_digits = '3.141592653589793238462643383279502884197169399375'
len(pi_digits)
float(pi_digits)

#6) #list
numbers = [2, 1, 3, 5, 8]
squares = (n**2 for n in numbers)
squares #objectgenerator
list_of_squares = list(squares)
list_of_squares
copy_of_squares = list_of_squares.copy() #use if you know you're working with list
copy_of_squares = list(list_of_squares)  #use when you don't know what iterable is, general loop
my_list = []        #Do this for empty list generation

#7) #tuple
numbers = [2, 1, 3, 4, 7]
tuple(numbers) #like list, use tuple to make hashable collections - dict keys for example

#8) #dict
#Loops over an iterable of key-value pairs, makes dict
color_counts = [('red', 2), ('green', 1), ('blue', 3), ('purple', 5)]
#This
colors = {}
for color, n in color_counts:
    colors[color] = n
colors
#Becomes this
colors = dict(color_counts)
colors
#dict function accepts 2 args: another dictionary(mapping - dict is copied) or a list of k-v tuples(iterable, new dict constructed)
colors
new_dictionary = dict(colors)
new_dictionary
#Accepts kwars to make it string based keys
person = dict(name='Trey Hunner', profession='Python Trainer')
person
person = {'name': 'Trey Hunner', 'profession': 'Python Trainer'} #literal, best way to create dict
person
my_list = {}   #Do this to create empty dict

#9) #set
#takes an iterable of hashable values(str, nums, or other immutable types) and returns a set
numbers = [1, 1, 2, 3, 5, 8]
set(numbers)
numbers = set()  #Way to make empty set
#Asterisks in List Literals (https://treyhunner.com/2018/10/asterisks-in-python-what-they-are-and-how-to-use-them/#Asterisks_in_list_literals)
fruits = ['lemon', 'pear', 'watermelon', 'tomato']
(*fruits[1:], fruits[0])    #('pear', 'watermelon', 'tomato', 'lemon')

#10) #range
#gives a range object, which represents a range of numbers
range(10_000)   
range(-1_000_000_000, 1_000_000_000)
#Range includes start number, but excludes stop number range(0,10) does not include 10!
for n in range(0, 50, 10):  #useful for loops
    print(n)
#common use case is to do operation n times (list comprehension)
#tool to transform one list(any iterable) into another list -filter followed by a map
numbers = [1, 2, 3, 4, 5]

doubled_odds = []
for n in numbers:
    if n % 2 == 1:
        doubled_odds.append(n * 2)
#Becomes
doubled_odds = [n * 2 for n in numbers if n % 2 == 1]
def get_things():
    pass
first_five = [get_things() for _ in range(5)]



###Useful Built-ins Functions Overlooked by Pythonistas####
#1) #bool
#check truthiness of a python object. For numbers, truth= non-zero
bool(5) #True   
bool(0) #False
#For collections, truthiness is usually a question of non-emptyness (len>0)
bool('hello')   #True
bool('')        #False
bool(['a'])     #True
bool([])        #False
bool({})        #False
bool({1:1, 3:9})#True
bool(None)      #False
#Truth Valuse Testing - used like:
if not numbers:
    print("The numbers list is empty")

#2) #enumerate
#Count upward one number at a time while looping over an iterable
#example tracking line number in a file
"""
with open('hello.txt', mode='rt') as my_file:
    for n, line in enumerate(my_file, start=1):
        print(f"{n:03}", line)
"""
#Also used to track the index of items in a sequence
def palindromic(sequence):
    """Return True if the sequence is the same thing in reverse."""
    for i, item in enumerate(sequence):
        if item != sequence[-(i+1)]:
            return False
    return True
#use enumerate instead of range(len(sequence))

#3) #zip
#specialized function to loop over multiple iterables at the same time
one_iterable = [2, 1, 3, 4, 7, 11]
another_iterable = ['P', 'y', 't', 'h', 'o', 'n']
for n, letter in zip(one_iterable, another_iterable):
    print(letter, n)
#for loop over two lists at the same time, enumerate when you need indexes
#for zip on iterables of different lengths, look at: itertools.zip_longest

#4) #reversed
numbers = [2, 1, 3, 4, 7]
reversed(numbers)   #returns an iterator object
#Can loop over the iterator once:
reversed_numbers = reversed(numbers)
list(reversed_numbers)  #[7,4,3,1,2]
list(reversed_numbers)  #[]
#reversed serves best as a Looping helper function - in for part of loop:
for n in reversed(numbers):
    print(n)
#Other ways to reverse in python:
#for n in numbers[::-1]:    #slicing (builds new list)
#numbers.reverse()          #in-place method (mutates list)
#Reversed usually best way to reverse any iterable in Python, lazy iterator retrieves as we loop
#Example non-copy function w/ reversed and zip - no copying of lists is done here
def palindromic(sequence):
    """Return True if the sequence is the same thing in reverse."""
    for n, m in zip(sequence, reversed(sequence)):
        if n != m:
            return False
    return True

#5) #sum
#takes iterable of numbers and returns the sum of those numbers
sum([2, 1, 3, 4, 7])    #17
#helper func that does the looping for you, pair nice with generator expressions:
numbers = [2, 1, 3, 4, 7, 11, 18]
sum(n**2 for n in numbers)  #524

#6) #min and max
numbers = [2, 1, 3, 4, 7, 11, 18]
min(numbers)    #1
max(numbers)    #18
#Compare by using < operator, allow for key function to customize min/max meaning

#7) #sorted
numbers = [1, 8, 2, 13, 5, 3, 1]
words = ["python", "is", "lovely"]
sorted(words)                   #['is', 'lovely', 'python']
sorted(numbers, reverse=True)   #[13, 8, 5, 3, 2, 1, 1]
#like min/max, compares with <, can sey custom key function
#use list.sort for lists(lose original), use sorted() for other iterables

#8) #any and all
#pair with generator expression to determin whether any/all items in iterable match given condition
def palindromic(sequence):
    """Return True if the sequence is the same thing in reverse."""
    return all(             #return not any(
        n == m              #n != m
        for n, m in zip(sequence, reversed(sequence))
    )   #Can rewrite palindromic func, returns bool



####5 Debugging Functions#####
#1) #breakpoint 
#Can pause execution of code, drop into a python command propt(pdb)
#equivalent prior to python3.7 : import pdb;   pdb.set_trace()

#2) #dir
#Use One - see a list of all your local variables
#Use Two - see a list of all attributes on a particular object
dir()   #['__annotations__', '__doc__', '__name__', '__package__']
x = [1, 2, 3, 4]
dir()   #['__annotations__', '__doc__', '__name__', '__package__', 'x']
dir(x)  #['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
#x object variable shows typical list methods - append, pop, remove

#3) #vars
#Checks locals() and tests the __dict__ attribute of objects
#When called w/ no args, == calling locals() built in(sows dict of all local vars and values)
vars()  #{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <class '_frozen_importlib.BuiltinImporter'>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>}
#When called w/ args, accesses __dict__ attribute on object:
from itertools import chain
vars(chain) #mappingproxy({'__getattribute__': <slot wrapper '__getattribute__' of 'itertools.chain' objects>, '__iter__': <slot wrapper '__iter__' of 'itertools.chain' objects>, '__next__': <slot wrapper '__next__' of 'itertools.chain' objects>, '__new__': <built-in method __new__ of type object at 0x5611ee76fac0>, 'from_iterable': <method 'from_iterable' of 'itertools.chain' objects>, '__reduce__': <method '__reduce__' of 'itertools.chain' objects>, '__setstate__': <method '__setstate__' of 'itertools.chain' objects>, '__doc__': 'chain(*iterables) --> chain object\n\nReturn a chain object whose .__next__() method returns elements from the\nfirst iterable until it is exhausted, then elements from the next\niterable, until all of the iterables are exhausted.'})

#4) #type
x = [1, 2, 3]
type(x)     #<class 'list'>
type(list)  #<class 'type'>
#tells type of obj you pass
#Use type instead of __class__:
x.__class__
type(x)         #both list
#Type can be helpful in OOC with inheritance and custom str representations. Also debugging use
#Use isinstance function for type checking

#5) #help
#Aids debugging to know how certain object, method, or attribute works
#help(list.insert) vs looking up the method docs via Google



###Function to Know of, Learn it Later#####
#Lot of built-ins to know of, but may not need right now. Learn when you need to use them!
#1) #open
#open files to read and write to. Also know pathlib is alternative that may be better

#2) #input
#Prompts user for input, waits until Enter key is hit, returns text they typed
#Reads from standard input. Alternative is CL args, read from config file, read from DB

#3) #repr
#Programmer-readable representation of an object
#str and repr are the same for many objects:
str(4), repr(4)     #('4', '4')
str([]), repr([])   #('[]', '[]')
#str and repr different for others:
str('hello'), repr("hello")     #('hello', "'hello'")
from datetime import date
str(date(2020, 1, 1)), repr(date(2020, 1, 1))   #('2020-01-01', 'datetime.date(2020, 1, 1)')
#used when logging, handling exceptions, and implementing dunder methods

#4) #super
#Needed when inheriting from another Python class
#Django users and class makers will learn it; Python rarely create classes

#5) #property
#Is a decorator(function returning another function, applied as @wrapper) and a descriptor(defines methods __get__(), __set__(), __delete__())
#Creates an attribute which will always seem to contain the return value of a particular function call. Example class w/ property:
class Circle:

    def __init__(self, radius=1):
        self.radius = radius

    @property
    def diameter(self):
        return self.radius * 2
#Access of that diameter attribute on a Circle Object:
circle = Circle()
circle.diameter #2
circle.radius = 5
circle.diameter #10
#Use properties instead of getter methods and setter methods

#6) #issubclass and isinstance
#issubclass checks whether a class is a subclass of one or more other classes
issubclass(int, bool)   #False
issubclass(bool, int)   #True
issubclass(bool, object)#True
#isinstance checks whether an object is an instance of one or more classes
isinstance(True, str)   #False
isinstance(True, bool)  #True
#Think of is instance as delgating to issubclass:
issubclass(type(True), bool)    #True

#7) #hasattr, getattr, setattr, delattr
#used to work with an attribute on an object but the attribute name is dynamic
class Thing: pass
thing = Thing()
hasattr(thing, 'x')     #False; check if obj has certain attr
thing.x = 4
hasattr(thing, 'x')     #True
getattr(thing, 'x', 0)  #4; Retrieve value of attr(optional default if it doesn't exist)
setattr(thing, 'x', 5)
thing.x                 #5 ;set value
delattr(thing, 'x')
thing.x                 #error no attribute

#8) #classmethod and staticmethod
#Decorators. If you have a a method that should be callable on either an instance or a class, want classmethod. Ex. factory method use:
class RomanNumeral:

    """A Roman numeral, represented as a string and numerically."""

    def __init__(self, number):
        self.value = number

    @classmethod
    def from_string(cls, string):
        return cls(cls.roman_to_int(string))  # function doesn't exist yet
    
    @staticmethod
    def roman_to_int(numeral):
        total = 0
        for symbol, next_symbol in zip_longest(numeral, numeral[1:]):
            value = RomanNumeral.SYMBOLS[symbol]
            next_value = RomanNumeral.SYMBOLS.get(next_symbol, 0)
            if value < next_value:
                value = -value
            total += value
        return total
#roman_to_int function doesn't require access to the instane or the class

#9) #next
#next returns the next item in an iterator
#common iterators: enumerate objects, zip objects, return value of the reversed function, files(from open function), csv.reader objects, generator expressions, generator functions
#think of next as a way to manually loop over an iterator to get a single item and then break
numbers = [2, 1, 3, 4, 7, 11]
squares = (n**2 for n in numbers)
next(squares)   #4
for n in squares:
    break
n               #1
next(squares)   #9



###Maybe Learn it Eventually, More Specialized Functions####
"""
-iter: get an iterator from an iterable: this function powers for loops and it can be very useful when you’re making helper functions for looping lazily
-callable: return True if the argument is a callable (I talked about this a bit in my article functions and callables)
-filter and map: as I discuss in my article on overusing lambda functions, I recommend using generator expressions over the built-in map and filter functions
-id, locals, and globals: these are great tools for teaching Python and you may have already seen them, but you won’t see these much in real Python code
-round: you’ll look this up if you need to round a number
-divmod: this function does a floor division (//) and a modulo operation (%) at the same time
-bin, oct, and hex: if you need to display a number as a string in binary, octal, or hexadecimal form, you’ll want these functions
-abs: when you need the absolute value of a number, you’ll look this up
-hash: dictionaries and sets rely on the hash function to test for hashability, but you likely won’t need it unless you’re implementing a clever de-duplication algorithm
-object: this function (yes it’s a class) is useful for making unique default values and sentinel values, if you ever need those
"""

###Likely Don't Need These Functions####
"""
-ord and chr: these are fun for teaching ASCII tables and unicode code points, but I’ve never really found a use for them in my own code
-exec and eval: for evaluating a string as if it was code
-compile: this is related to exec and eval
-slice: if you’re implementing __getitem__ to make a custom sequence, you may need this (some Python Morsels exercises require this actually), but unless you make your own custom sequence you’ll likely never see slice
-bytes, bytearray, and memoryview: if you’re working with bytes often, you’ll reach for some of these (just ignore them until then)
-ascii: like repr but returns an ASCII-only representation of an object; I haven’t needed this in my code yet
-frozenset: like set, but it’s immutable (and hashable!); very neat but not something I’ve needed in my own code
-__import__: this function isn’t really meant to be used by you, use importlib instead
-format: this calls the __format__ method, which is used for string formatting (f-strings and str.format); you usually don’t need to call this function directly
-pow: the exponentiation operator (**) usually supplants this… unless you’re doing modulo-math (maybe you’re implementing RSA encryption from scratch…?)
-complex: if you didn’t know that 4j+3 is valid Python code, you likely don’t need the complex function
"""

#Extra Notes
#operator.itemgetter and operator.attrgetter #https://docs.python.org/3/library/operator.html
#dict().setdefault()
#python 3.10 Pattern Matching (switches)  #https://peps.python.org/pep-0636/
#f-strings -- .strip(), .split(), .join()
#f”{x:,}” to return the value of x as a comma separated number string
#functools partial()
