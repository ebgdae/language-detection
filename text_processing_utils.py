import collections
import pandas as pd
from typing import List, Tuple

def get_all_ngram_counts(text_sample: str, n: int) -> collections.Counter:
    """
    get_all_ngram_counts Get the frequency of each n-gram in the given text sample.
    The terms used for n-gram generation are individual characters.

    Parameters
    ----------
    text_sample : str
        the text sample to use to find n-grams
    n : int
        n for the n-gram

    Returns
    -------
    collections.Counter
        the frequency of each n-gram
    """

    # pad the sample with blank spaces
    padding_string = " "*(n-1)
    text_sample = padding_string + text_sample + padding_string
    text_sample = text_sample.lower()
    ngram_frequencies = collections.Counter()

    for i in range(len(text_sample) - len(padding_string)):
        ngram = text_sample[i:i+n]
        ngram_frequencies[ngram] += 1
    return ngram_frequencies


def create_ngram_dictionaries_for_language(text_samples: pd.DataFrame, n_values: List[int], ngrams_per_dictionary: int) -> List[collections.Counter]:
    """
    create_ngram_dictionaries_for_language Create a dictionary of n-grams for each value of n using the text samples given.
    A 'dictionary' is the ngrams_per_dictionary most frequently occuring ngrams in all the text samples

    Parameters
    ----------
    text_samples : pd.DataFrame
        A pandas dataframe containing a column 'Text'
        Each row of this column is a text sample of the same language

    n_values: List(int):
        The values of 'n' to use to create n-grams
    
    ngrams_per_dictionary : int
        The size of each dictionary

    Returns
    -------
    list(collections.Counter)
        List of dictionaries for each value of n
    """
    language_dictionaries = []
    for i in n_values:
        n_dictionary = collections.Counter()
        for sample in text_samples.Text:
            n_dictionary += get_all_ngram_counts(sample, i)

        n_dictionary = n_dictionary.most_common(ngrams_per_dictionary)
        language_dictionaries.append(n_dictionary)
    return language_dictionaries


def get_distribution_distance(language_dict: List[Tuple[str, int]], sample_dict: List[Tuple[str, int]]) -> int:
    """
    get_distribution_distance A distance metric between a language ngram frequency distribution and 
    a text sample ngram frequency distribution

    Parameters
    ----------
    language_dict : List
        eg. [('a',10), ('e',4), ('b',2)]
    sample_dict : List
        same structure as language_dict

    Returns
    -------
    int
        a distance measure
    """
    distance = 0
    for i in range(len(sample_dict)):
        present = False
        for j in range(len(language_dict)):
            if sample_dict[i][0] == language_dict[j][0]:
                present = True
                distance += abs(i-j)
        if not present:
            distance += len(sample_dict)-i
    return distance
    