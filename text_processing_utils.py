import collections
import pandas as pd
from typing import List, Tuple, Dict


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


def create_train_validate_test_sets(df : pd.DataFrame, train_size: float, validate_size: float, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    create_test_train_validate_sets 

    Parameters
    ----------
    df : pd.DataFrame
        
    train_size : float
        Training set size (in percentage, 0-100)
    test_size : float
        
    validate_size : float
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        training, validation, and test sets
    """
    assert train_size + test_size + validate_size == 100.0, 'the three set sizes should add up to 100%'

    train_df = pd.DataFrame(columns=df.columns)
    validate_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    languages = df.language.unique()

    for l in languages:
        language_df = df[df.language == l]
        language_train = language_df.sample(frac=train_size/100)
        test_and_validate = language_df.drop(language_train.index)
        language_validate = test_and_validate.sample(frac=validate_size/(validate_size + test_size))
        language_test = test_and_validate.drop(language_validate.index)

        train_df = train_df.append(language_train)
        validate_df = validate_df.append(language_test)
        test_df = test_df.append(language_test)

    return (train_df, validate_df, test_df)


def predict_language(text_sample: str, dictionaries: Dict, ngram_values: List[int], ngrams_per_dictionary: int) -> str:
    """
    predict_language predicts a language for a text sample

    Parameters
    ----------
    text_sample : str
        the text sample to predict
    dictionaries : Dict
        all dictionaries for each language
    ngram_values : List[int]
        the ngram values to use. If more than one ngram value, the mean distance is used
    ngrams_per_dictionary : int
        The size of each dictionary
    Returns
    -------
    str
        language
    """
    sample_dictionaries = []
    for i in ngram_values:
        n_dictionary = get_all_ngram_counts(text_sample, i)
        n_dictionary = n_dictionary.most_common(ngrams_per_dictionary)
        sample_dictionaries.append(n_dictionary)
    
    language_distances = []
    for language, language_dicts in dictionaries.items():
        language_distance = 0
        for i, val in enumerate(ngram_values):
            language_distance += get_distribution_distance(language_dicts[val-1], sample_dictionaries[i])
        language_distances.append((language, language_distance))
    
    closest_language = min(language_distances, key = lambda t:t[1])
    return closest_language[0]

