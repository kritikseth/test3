import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

def add_typing_error(sentence):
    key_neighbors = {
        'a': ['q', 's'], 'b': ['v', 'g'], 'c': ['x', 'd'], 'd': ['s', 'f'],
        'e': ['w', 'r'], 'f': ['d', 'g'], 'g': ['f', 'h'], 'h': ['g', 'j'],
        'i': ['u', 'o'], 'j': ['h', 'k'], 'k': ['j', 'l'], 'l': ['k', 'o'],
        'm': ['n', 'k'], 'n': ['b', 'm'], 'o': ['i', 'p'], 'p': ['o', 'l'],
        'q': ['a', 'w'], 'r': ['e', 't'], 's': ['a', 'd'], 't': ['r', 'y'],
        'u': ['y', 'i'], 'v': ['c', 'b'], 'w': ['q', 'e'], 'x': ['z', 'c'],
        'y': ['t', 'u'], 'z': ['x', 's']
    }

    words = word_tokenize(sentence)
    transformed_words = []

    for word in words:
        if random.random() <= 0.1 and len(word) > 2 and word not in stopwords:
            for _ in range(random.randint(1, 2)):
                index = random.randint(0, len(word) - 1)
                original_letter = word[index]
                if original_letter.isalpha():
                    replacement_letter = random.choice(key_neighbors.get(original_letter, [original_letter]))
                    word = word[:index] + replacement_letter + word[index + 1:]

        transformed_words.append(word)

    return " ".join(transformed_words)

def replace_words_with_synonyms(sentence):
    words = word_tokenize(sentence)
    transformed_words = []

    num_replacements = random.randint(1, 2)
    replacements_made = 0

    for word in words:
        if replacements_made < num_replacements and word not in stopwords:
            if random.random() <= 0.3:
                synonyms = get_synonyms(word)
                if synonyms:
                    transformed_words.append(random.choice(synonyms))
                    replacements_made += 1
                else:
                    transformed_words.append(word)
            else:
                transformed_words.append(word)
        else:
            transformed_words.append(word)

    return " ".join(transformed_words)

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

def replace_preposition(sentence):

    prepositions = {
        "at": ["in", "on", "to", "for"],
        "in": ["into", "at", "on", "to",  "for"],
        "on": ["onto", "at", "in", "to", "for"],
        "to": ["at", "in", "on", "for"],
        "for": ["at", "in", "on", "to"]
    }

    words = word_tokenize(sentence)
    transformed_words = []
    for word in words:
        if word.lower() in prepositions:
            replacement = random.choice(prepositions[word.lower()])
            transformed_words.append(replacement)
        else:
            transformed_words.append(word)
    
    return " ".join(transformed_words)

def replace_or_remove_articles(sentence):
    articles = ["a", "an", "the"]
    words = word_tokenize(sentence)
    transformed_words = []

    for word in words:
        if word.lower() in articles:
            if random.random() <= 0.5:
                transformed_words.append("")
            else:
                incorrect_articles = [a for a in articles if a != word.lower()]
                replacement = random.choice(incorrect_articles)
                transformed_words.append(replacement)
        else:
            transformed_words.append(word)
    
    return " ".join(transformed_words)

def remove_be_forms(sentence):
    be_forms = ["am", "is", "are", "was", "were"]
    words = word_tokenize(sentence)
    transformed_words = []

    for word in words:
        if word.lower() in be_forms:
            transformed_words.append("")
        else:
            transformed_words.append(word)
    
    return " ".join(transformed_words)

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    ###
    sentences = example["text"].split(".")
    transformed_sentences = []

    for sentence in sentences:
        # Replace words with synonyms and introduce typing errors
        transformed_sentence = replace_words_with_synonyms(sentence)
        transformed_sentence = add_typing_error(transformed_sentence)

        # Apply additional transformations with a 75% probability
        if random.random() <= 0.50:
            # Randomly choose which additional transformations to apply
            additional_transformations = [replace_preposition, replace_or_remove_articles, remove_be_forms]
            selected_transformations = random.sample(additional_transformations, random.randint(3, 3))

            for transformation in selected_transformations:
                transformed_sentence = transformation(transformed_sentence)

        transformed_sentences.append(transformed_sentence)

    # Reconstruct the text with transformed sentences
    example["text"] = ". ".join(transformed_sentences)
    ##### YOUR CODE ENDS HERE ######

    return example
