from game import *
from functools import cache
from collections import defaultdict

@cache
def minimax(state):
    if not state:
        return 0
    elif len(state) == 1:
        return 1

    # the least number of rounds player needed to win
    best_cost = float("inf")

    for i in range(len(valid_words)):
        guess = valid_words[i]

        # partition all answers into baskets, according to the feedback
        baskets = defaultdict(list)
        for answer in state:
            baskets[ternary(compare(guess, answer))].append(answer)

        # only one basket means no information gained, so skip current guess
        if len(baskets) == 1:
            continue

        # by host choosing feedbacks, the largest number of rounds player needed to win
        worst_cost = 0

        for basket in baskets.values():
            next_state = tuple(basket)
            cost = minimax(next_state)
            worst_cost = max(worst_cost, cost)

            # if the following condition happens, our current guess can't improve best_cost
            # since we take the maximum among all worst_cost, we know
            # 1 + max(all worst cost) >= best_cost
            # after that, best_cost = min(best_cost, 1 + worst_cost) will not change best_cost
            # additionally, even if we break early, best_cost still won't change
            # since we already observed a worst_cost such that 1 + worst_cost >= best_cost
            if 1 + worst_cost >= best_cost:
                break

        best_cost = min(best_cost, 1 + worst_cost)

    return best_cost

@cache
def minimax_full(state):
    if not state:
        return 0, None, None
    elif len(state) == 1:
        return 1, state[0], (2, 2, 2, 2, 2)

    # the least number of rounds player needed to win
    best_cost = float("inf")
    best_color = None
    best_guess = None

    for i in range(len(valid_words)):
        guess = valid_words[i]

        # partition all answers into baskets, according to the feedback
        baskets = defaultdict(list)
        for answer in state:
            baskets[ternary(compare(guess, answer))].append(answer)

        # only one basket means no information gained, so skip current guess
        if len(baskets) == 1:
            continue

        # by host choosing feedbacks, the largest number of rounds player needed to win
        worst_cost = 0
        # host's best response to guess
        guess_color = None

        for color_id, basket in baskets.items():
            next_state = tuple(basket)
            cost = minimax_full(next_state)[0]

            if cost > worst_cost:
                worst_cost = cost
                guess_color = colorings[color_id]

            # if the following condition happens, our current guess can't improve best_cost
            # since we take the maximum among all worst_cost, we know
            # 1 + max(all worst cost) >= best_cost
            # after that, best_cost = min(best_cost, 1 + worst_cost) will not change best_cost
            # additionally, even if we break early, best_cost still won't change
            # since we already observed a worst_cost such that 1 + worst_cost >= best_cost
            if 1 + worst_cost >= best_cost:
                break

        if 1 + worst_cost < best_cost:
            best_cost = 1 + worst_cost
            best_guess = guess
            best_color = guess_color

    return best_cost, best_guess, best_color

@cache
def absurdle_optimal(state):
    if not state:
        return 0
    elif len(state) == 1:
        return 1

    best_cost = float("inf")

    for guess in valid_words:

        baskets = defaultdict(list)
        for answer in state:
            baskets[ternary(compare(guess, answer))].append(answer)

        # No information gained
        if len(baskets) == 1:
            continue

        # Absurdle chooses the largest basket
        # with its tie-breaking rules
        best_color = min(
            baskets,
            key=lambda i: (
                -len(baskets[i]),
                colorings[i].count(2),
                colorings[i].count(1),
                i
            )
        )

        max_basket = baskets[best_color]

        # Recursively calculate the optimal cost
        cost = 1 + absurdle_optimal(tuple(max_basket))

        best_cost = min(best_cost, cost)

        # At least 2 guesses are needed when len(state) >= 2
        if best_cost == 2:
            break

    return best_cost
