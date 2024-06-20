import dianna
import math
import random

import numpy as np

from dianna.methods.lime import LIMEImage, LIMEText
from onnx.onnx_ml_pb2 import ModelProto
from numpy.typing import NDArray
from typing import Union, Optional, Callable
from torchtext.vocab import Vectors

from dianna.utils.tokenizers import SpacyTokenizer


def get_onnx_names(onnx_model: ModelProto) -> tuple:
    '''
    Gets the names of the input and output layers used to save an onnx model.

    Args:
        onnx_model: The model to extract the names out of.
    Returns:
        net_feed_input, output: names used for the input and output layers.
    '''
    output =[node.name for node in onnx_model.graph.output]
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))
    return net_feed_input, output


def LIME_postprocess(modality, *args, **kwargs) -> NDArray:
    '''
    Post-process the output of DIANNA LIME in according to what Quantus expects. 

    DIANNA yields: list[NDArray[(Any, Any), Any]]
    Quantus expects: NDArray((Any, Any, Any), Any)
    '''
    if modality == 'image':
        init_kwargs = dianna.utils.get_kwargs_applicable_to_function(LIMEImage.__init__, 
                                                                    kwargs)
        explainer = LIMEImage(**init_kwargs)
        results = explainer.explain(*args, method='LIME', return_masks=False, **kwargs)
    elif modality == 'text':
        init_kwargs = dianna.utils.get_kwargs_applicable_to_function(LIMEText.__init__, 
                                                            kwargs)
        explainer = LIMEText(**init_kwargs)
        results = explainer.explain(*args, method='LIME', **kwargs)[0]
        results = [[x[-1] for x in results]]
    return np.array(results)


def SHAP_postprocess(*args, **kwargs) -> NDArray:
    '''
    Post-process the output of DIANNA KernelSHAP in according to what Quantus expects. 

    DIANNA yields: tuple[NDArray[(Any, Any), Any], NDArray[(Any, Any), Any]]]
    Quantus expects: NDArray((Any, Any, Any), Any)
    '''
    shapley_values, segments_slic = dianna.explain_image(method='KernelSHAP', 
                                                         axis_labels=('height','width','channels'),
                                                         *args, **kwargs)
    saliences = _fill_segmentation(shapley_values[0].flatten(), segments_slic)
    return saliences


def RISE_postprocess(modality, *args, **kwargs) -> NDArray:
    '''
    Post-process the output of DIANNA LIME in according to what Quantus expects. 

    DIANNA yields: list[NDArray[(Any, Any), Any]]
    Quantus expects: NDArray((Any, Any, Any), Any)
    '''
    if modality == 'image':
        results = dianna.explain_image(*args, method='RISE', **kwargs)
    elif modality == 'text':
        results = dianna.explain_text(method='RISE', *args, **kwargs)[0]
        results = [[x[-1] for x in results]]
    return np.array(results)


def _fill_segmentation(values: NDArray, segmentation: NDArray) -> NDArray:
    '''
    Helper function to mask a segmentation with Shapeley Values

    Args:
        values: Shapeley values
        segmentation: the indices where the shapeley values reside
    Returns:
        The segmented Shapeley values
    '''
    out = np.zeros(segmentation.shape)
    for i in range(segmentation.min(), segmentation.max()):
        out[segmentation == i] = values[i - segmentation.min()]
    return out[np.newaxis, ...]


class Synonym_replacer:
    def __init__(self):
        from nltk.corpus import wordnet
        
        self.wordnet = wordnet
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
                           'ours', 'ourselves', 'you', 'your', 'yours', 
                           'yourself', 'yourselves', 'he', 'him', 'his', 
                           'himself', 'she', 'her', 'hers', 'herself', 
                           'it', 'its', 'itself', 'they', 'them', 'their', 
                           'theirs', 'themselves', 'what', 'which', 'who', 
                           'whom', 'this', 'that', 'these', 'those', 'am', 
                           'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                           'have', 'has', 'had', 'having', 'do', 'does', 'did',
                           'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                           'because', 'as', 'until', 'while', 'of', 'at', 
                           'by', 'for', 'with', 'about', 'against', 'between',
                           'into', 'through', 'during', 'before', 'after', 
                           'above', 'below', 'to', 'from', 'up', 'down', 'in',
                           'out', 'on', 'off', 'over', 'under', 'again', 
                           'further', 'then', 'once', 'here', 'there', 'when', 
                           'where', 'why', 'how', 'all', 'any', 'both', 'each', 
                           'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                           'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
                           'very', 's', 't', 'can', 'will', 'just', 'don', 
                           'should', 'now', '']

    def __call__(self, arr, aug_p=.1, **kwargs):
        n = max(1, math.floor(len(arr) * aug_p))
        new_words = arr.copy()
        random_word_list = list(set([word for word in arr if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                
                new_words = [synonym if word == random_word else word for word in new_words]
                #print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= n: #only replace up to n words
                break

        #this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def get_synonyms(self, word):
        synonyms = set()
        for syn in self.wordnet.synsets(word): 
            for lemma in syn.lemmas(): 
                if '-' in lemma.name() or '_' in lemma.name():
                    continue
                synonym = lemma.name().lower()
                synonym = "".join([char for char in synonym if char.isalpha()]) 
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)


class Embedder:
    def __init__(self, vocab: Vectors, max_filter_size: Optional[int] = 5):
        self.vocab = vocab
        self.max_filter_size = max_filter_size

    def __call__(self, tokens):
        if isinstance(tokens, (np.ndarray, np.generic)):
            tokens = tokens.tolist()
        # pad if needed
        if self.max_filter_size is not None:
            tokens = self.pad(tokens, self.max_filter_size)
        # embed
        tokens = [self.vocab.stoi[token] if token in self.vocab.stoi 
                  else self.vocab.stoi['<unk>'] for token in tokens]
        return tokens
    
    def pad(self, tokens, padding_token='<pad>'):
        npad = max(self.max_filter_size - len(tokens), 0)
        tokens += [padding_token] * npad
        return tokens
        