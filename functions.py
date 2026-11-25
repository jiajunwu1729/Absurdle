from functools import cache

def load_wordlist(filepath: str) -> list[str]:
    """Load words from file, convert to uppercase"""
    with open(filepath, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]

def plus_one(nums: list[int]) -> list[int]:
    if nums[-1] == 2:
        return plus_one(nums[:-1]) + [0]
    else:
        nums[-1] += 1
        return nums

def generate() -> list[list[int]]:
    """Generate a list containing all possible wordle colorings"""
    result = []
    curr_list = [0, 0, 0, 0, 0]
    while curr_list != [2, 2, 2, 2, 2]:
        result.append(curr_list[:])
        curr_list = plus_one(curr_list)
    result.append([2, 2, 2, 2, 2])
    return result

@cache
def compare(guess: str, answer: str) -> list[int]:
    """Return Wordle coloring as list of 5 values: 0=grey, 1=yellow, 2=green"""
    result = [0] * 5
    guess = list(guess)
    answer = list(answer)

    # Green
    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = 2
            guess[i] = '#'
            answer[i] = '$'

    # Yellow
    for i in range(5):
        for j in range(5):
            if guess[i] == answer[j]:
                result[i] = 1
                answer[j] = '$'
                break

    return result

def reduce_possibilities(possible_answers, guess, color):
    result = []
    for answer in possible_answers:
        if compare(guess, answer) == color:
            result.append(answer)
    return result

def ternary(color: list[int]) -> int:
    """Convert each wordle coloring to a unique number by ternary"""
    return sum([color[i] * 3**(4 - i) for i in range(5)])