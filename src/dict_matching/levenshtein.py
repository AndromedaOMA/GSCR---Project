import os
import sys
import unicodedata


# Checks if two passed letters are a diacritic and a non-diacritic version of the same letter
# i.e. (s, ș)
def is_base_letter_and_diacritic_form(char1, char2):
    def get_base_and_diacritics(char):
        normalized = unicodedata.normalize('NFD', char)
        base = ''
        diacritics = []
        for c in normalized:
            if unicodedata.category(c).startswith('M'):
                diacritics.append(c)
            else:
                base += c
        return base, diacritics

    base1, diacritics1 = get_base_and_diacritics(char1)
    base2, diacritics2 = get_base_and_diacritics(char2)
    # Case 1: char1 is base, char2 is precomposed or base + combining
    if len(base1) == 1 and unicodedata.category(base1).startswith('L') and len(base2) == 1 and unicodedata.category(
            base2).startswith('L'):
        return base1 == base2 and (char1 != char2)  # Different but same base
    # Case 2: One is base, the other is base + single combining mark
    if len(base1) == 1 and unicodedata.category(base1).startswith('L') and len(diacritics2) == 1 and base1 == base2:
        # Check if combining char2 with base1 results in char2
        return unicodedata.normalize('NFC', base1 + diacritics2) == char2
    if len(base2) == 1 and unicodedata.category(base2).startswith('L') and len(diacritics1) == 1 and base2 == base1:
        # Check if combining char1 with base2 results in char1
        return unicodedata.normalize('NFC', base2 + diacritics1) == char1
    return False


# Calculates the levenshtein distance between two words
def levenshtein(word1, word2):
    if not isinstance(word1, str) or not isinstance(word2, str):
        raise TypeError("Both passed arguments must be of type String")
    n = len(word2)
    m = len(word1)
    prev_row = [i for i in range(n + 1)]
    curr_row = [0] * (n + 1)
    for i in range(m):
        curr_row[0] = i + 1
        for j in range(n):
            deletion_cost = prev_row[j + 1] + 1
            insertion_cost = curr_row[j] + 1
            substitution_cost = prev_row[j]
            if word1[i] != word2[j]:
                if not is_base_letter_and_diacritic_form(word1[i], word2[j]):
                    substitution_cost += 1
                else:
                    substitution_cost += 0.25
            curr_row[j + 1] = min(deletion_cost, insertion_cost, substitution_cost)
        prev_row = curr_row
        curr_row = [0] * (n + 1)
    return prev_row[n]


# Function that loads the vocabulary used for testing
# Assumed to be file input (i.e. from .txt) but can be easily modified
# Return a dictionary scored as 0 for all words to be checked
def load_vocab(filename):
    try:
        vocabulary_scores = {}
        with open(filename, 'r', encoding='utf-8') as vocabulary_file:
            for word in vocabulary_file:
                vocabulary_scores[word.strip()] = 0
        return vocabulary_scores
    except FileNotFoundError:
        print(f"File {filename} not found")
    except Exception as e:
        print(f"An error occurred: {e}")


# Checks words in passed vocabulary against the word to be tested
# If any word is closer to the chosen word than the chosen accuracy
# Then it is passed in the returned list
def pattern_match_vocab(vocabulary, chosen_word, chosen_accuracy=3):
    potential_matches = []
    for word in vocabulary:
        try:
            vocabulary[word] = levenshtein(chosen_word, word)
            if vocabulary[word] == 0:
                return None  # If distance to vocab word is 0 then the word is correct so no need for further checks
            elif vocabulary[word] <= chosen_accuracy:
                potential_matches.append((word, vocabulary[word]))
        except TypeError:
            if not isinstance(chosen_word, str):
                print(f"Chosen word ({chosen_word}) provided not of type string")
            elif not isinstance(word, str):
                print(f"Provided vocabulary word {word} not of type string. Check vocabulary file for errors.")
    potential_matches.sort(key=lambda x: x[1])
    return potential_matches


# Example of usage with arguments passed at cli
# For testing mostly as obviously this will be called inside another program
# NOT CONFIGURED FOR INTEGRATION YET
if __name__ == "__main__":
    if len(sys.argv) == 3:
        chosen_word = sys.argv[1]
        filename = sys.argv[2]
        vocabulary = load_vocab(filename)
        print(pattern_match_vocab(vocabulary, chosen_word, 3))
