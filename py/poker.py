#! python3  - poker.py
# Peter Norvig Udacity poker
# https://www.youtube.com/watch?v=49qSOQpFExQ&list=PL096CA71DA6198753&index=1

import random


def poker(hands):
    "Return a list of winning hands: poker([hand, ...]) => [hand,...]"
    return max(hands, key=hand_rank)
    # return allmax(hands, key=hand_rank)


def allmax(iterable, key=None):
    "Return a list of all items equal to the max of the iterable."
    result, maxval = [], None
    key = key or (lambda x: x)
    for x in iterable:
        xval = key(x)
        if not result or xval > maxval:
            result, maxval = [x], xval
        elif xval == maxval:
            result.append(x)
    return result


def hand_rank(hand):
    "Return a value indicating how high the hand ranks."
    # counts is the count of each rank; ranks lists corresponding ranks
    # E.g. '7 T 7 9 7 => counts = (3, 1, 1); ranks = (7, 10, 9)
    groups = group(["--23456789TJQKA".index(r) for r, s in hand])
    counts, ranks = unzip(groups)
    if ranks == (14, 5, 4, 3, 2):
        ranks = (5, 4, 3, 2, 1)
    straight = len(ranks) == 5 and max(ranks) - min(ranks) == 4
    flush = len(set([s for r, s in hand])) == 1
    return (
        (
            9
            if (5,) == counts
            else 8
            if straight and flush
            else 7
            if (4, 1) == counts
            else 6
            if (3, 2) == counts
            else 5
            if flush
            else 4
            if straight
            else 3
            if (3, 1, 1) == counts
            else 2
            if (2, 2, 1) == counts
            else 1
            if (2, 1, 1, 1) == counts
            else 0
        ),
        ranks,
    )


"""           
    #Code to replace hand_rank long return - form dict approach that replaces
    return max(count_rankings[counts], 4*straight + 5*flush), ranks
    
count_rankings = {(5,):10, (4,1):7, (3,2):6, (3,1,1):3, (2,2,1):2,
                  (2,1,1,1):1, (1,1,1,1,1):0}
"""


def group(items):
    "Return a list of [(count, x)...], highest count first, then highest x first."
    groups = [(items.count(x), x) for x in set(items)]
    return sorted(groups, reverse=True)


def unzip(pairs):
    return zip(*pairs)


"""
def hand_rank(hand):
    "Return a value indicating how high the hand ranks."
    ranks = card_ranks(hand)
    if straight(ranks) and flush(ranks):
        return (8, max(ranks))
    elif kind(4, ranks):
        return (7, kind(4, ranks), kind(1, ranks))
    elif kind(3, ranks) and kind(2, ranks):
        return (6, kind(3, ranks), kind(2, ranks))
    elif flush(hand):
        return (5, ranks)
    elif straight(ranks):
        return (4, max(ranks))
    elif kind(3, ranks):
        return (3, kind(3, ranks), ranks)
    elif two_pair(ranks):
        return (2, two_pair(ranks), ranks)
    elif kind(2, ranks):
        return (1, kind(2, ranks), ranks)
    else:
        return (0, ranks)
"""


def card_ranks(hand):
    "Return a list of the ranks, sorted with higher first."
    ranks = ["--23456789TJQKA".index(r) for r, s in hand]
    ranks.sort(reverse=True)
    return [5, 4, 3, 2, 1] if (ranks == [14, 5, 4, 3, 2]) else ranks


def straight(ranks):
    "Returns True if the ordered ranks form a 5-card straight."
    return (max(ranks) - min(ranks) == 4) and len(set(ranks)) == 5


def flush(hand):
    "Return True if all the cards have the same suit."
    suits = [s for r, s in hand]
    return len(set(suits)) == 1


def kind(n, ranks):
    """Returns the first rank that this hand has exactly n of.
        Return None if there is no n-of-a-kind in the hand."""
    for r in ranks:
        if ranks.count(r) == n:
            return r
    return None


def two_pair(ranks):
    """ If there are two pair, return the two ranks as a
    tuple: (highest, lowest); otherwise return None."""
    pair = kind(2, ranks)
    lowpair = kind(2, list(reversed(ranks)))
    if pair and lowpair != pair:
        return (pair, lowpair)
    else:
        return None


def deal(numhands, n=5, deck=[r + s for r in "23456789TJQKA" for s in "SHDC"]):
    "Shuffle the deck and deal out numhands n-card hands."
    random.shuffle(deck)
    return [deck[n * i : n * (i + 1)] for i in range(numhands)]


def hand_percentages(n=700 * 1000):
    "Sample n random hands and print a table of percentages for each type of hand."
    counts = [0] * 9
    for i in range(n / 10):
        for hand in deal(10):
            ranking = hand_rank(hand)[0]
            counts[ranking] += 1
    for i in reversed(range(9)):
        print("%14s: %6.3f %%" % (hand[i], 100.0 * counts[i] / n))


def test():
    "Test cases for the functions in poker program."
    sf = "6C 7C 8C 9C TC".split()  # straight flush
    fk = "9D 9H 9S 9C 7D".split()  # four of a kind
    fh = "TD TC TH 7C 7D".split()  # full house
    tp = "5S 5D 9H 9C 6S".split()  # two pair
    s1 = "AS 2S 3S 4S 5C".split()  # A-5 straight
    s2 = "2C 3C 4C 5S 6S".split()  # 2-6 straight
    ah = "AS 2S 3S 4S 6C".split()  # A high
    sh = "2S 3S 4S 6C 7D".split()  # 7 high
    assert poker([s1, s2, ah, sh]) == s2
    fkranks = card_ranks(fk)
    tpranks = card_ranks(tp)
    assert kind(4, fkranks) == 9
    assert kind(3, fkranks) == None
    assert kind(2, fkranks) == None
    assert kind(1, fkranks) == 7
    assert two_pair(fkranks) == None
    assert two_pair(tpranks) == (9, 5)
    assert straight([9, 8, 7, 6, 5]) == True
    assert straight([9, 8, 8, 6, 5]) == False
    assert flush(sf) == True
    assert flush(fk) == False
    assert card_ranks(sf) == [10, 9, 8, 7, 6]
    assert card_ranks(fk) == [9, 9, 9, 9, 7]
    assert card_ranks(fh) == [10, 10, 10, 7, 7]
    assert poker([sf, fk, fh]) == sf
    assert poker([fk, fh]) == fk
    assert poker([fh, fh]) == fh
    assert poker([sf]) == sf
    assert poker([sf] + 99 * [fh]) == sf
    # assert hand_rank(sf) == (8, 10)
    # assert hand_rank(fk) == (7, 9, 7)
    # assert hand_rank(fh) == (6, 10, 7)

    return print("tests pass")


test()
