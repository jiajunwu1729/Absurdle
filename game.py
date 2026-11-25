from functions import *

valid_words = load_wordlist("valid-wordle-words.txt")
valid_answers = load_wordlist("valid-wordle-answers.txt")
colorings = generate()
boxes = ["⬜️", "🟨", "🟩️"]

class Game:

    def __init__(self, valid_answers=tuple(valid_answers)):
        self.words = tuple(valid_words)
        self.possible_answers = tuple(valid_answers)
        self.count = 0

    def find_coloring(self, guess):
        if not self.possible_answers:
            raise ValueError("No possible answers")

        frequency = [0 for _ in range(243)]
        for answer in self.possible_answers:
            frequency[ternary(compare(guess, answer))] += 1
        idx = min(
            range(243),
            key=lambda i: (
                -frequency[i],
                colorings[i].count(2),
                colorings[i].count(1),
                i
            )
        )
        return colorings[idx]

    def update_possible_answers(self, guess, color):
        self.possible_answers = tuple(
            reduce_possibilities(self.possible_answers, guess, color)
        )

    def play(self):
        while True:
            print("Possible answers:")
            print(self.possible_answers)
            guess = input("Enter your guess: ").upper()

            self.count += 1
            color = self.find_coloring(guess)
            print(list(map(lambda x: boxes[x], color)))
            if len(self.possible_answers) == 1 and guess == self.possible_answers[0]:
                print(f"You win! Number of tries: {self.count}")
                break

            self.update_possible_answers(guess, color)
