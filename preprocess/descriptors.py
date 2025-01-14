import numpy as np
import torch
import os
import re
import json
from sentence_transformers import SentenceTransformer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.add("product")
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import torch
import os
import re
import json
from textblob import TextBlob

class TextEmbedModel:
    """
    Class for the Word2Vec model.

    """

    def __init__(self, embed_model_name="glove-wiki-gigaword-300"):
        """
        Initialise a new text embbedding model.

        :param embed_model_name: the name of the word2vec model
        """
        self.embed_model_name = embed_model_name
        path = self.embed_model_name + ".model"
        if os.path.exists(path):
            self.model = KeyedVectors.load(path)
        else:
            self.model = api.load(self.embed_model_name)
            self.model.save(path)

        # Embed size specifies the embedding size and is also the hidden size
        # of the first hidden layer of memory cells.
        self.embed_size = int(self.embed_model_name.split("-")[-1])

    def get_vector_rep(self, phrase):
        """
        Get the vector representation of phrase.

        :param phrase: a phrase (i.e., words separated by a space)
        :return: the vector representation of phrase based on the word2vec model
        initialised
        """
        phrase = phrase.strip().lower()
        blob = TextBlob(phrase)
        words = blob.words

        vec = np.zeros([self.embed_size])

        for word in words:
            try:
                vec += self.model.get_vector(word)
            except KeyError:
                print("DOES NOT EXIST", word)
                vec += np.zeros([self.embed_size])

        return vec

class SentenceEmbedModel:
    """
    Class for the Word2Vec model.

    """

    def __init__(self, embed_model_name='sentence-transformers/average_word_embeddings_glove.6B.300d'):
        """
        Initialise a new text embbedding model.

        :param embed_model_name: the name of the word2vec model
        """
        self.embed_model_name = embed_model_name
        path = "./average_word_embeddings_glove.6B.300d"
        if os.path.exists(path):
            self.model = SentenceTransformer(path)
        else:
            self.model = SentenceTransformer(embed_model_name)
            self.model.save(path)

        # Embed size specifies the embedding size and is also the hidden size
        # of the first hidden layer of memory cells.
        self.embed_size = int(self.embed_model_name.strip("d").split(".")[-1])

    def get_vector_rep(self, phrase):
        """
        Get the vector representation of phrase.

        :param phrase: a phrase (i.e., words separated by a space)
        :return: the vector representation of phrase based on the sBERT model
        initialised
        """
        phrase = phrase.strip().lower()
        vec = np.zeros([self.embed_size])
        
        try:
            vec += self.model.encode(phrase)
        except KeyError:
            print("DOES NOT EXIST", phrase)
            vec += np.zeros([self.embed_size])

        return vec


class SentimentPreProcessor:
    """
    Class to pre-process the sentiments provided in the PITTs dataset.

    """

    def __init__(self, root="./data/annotations",
                 embed_model="glove-wiki-gigaword-300"):
        self.text_embed_model = TextEmbedModel(embed_model)
        self.embed_model = self.text_embed_model.embed_model_name
        self.model = self.text_embed_model.model
        self.embed_size = self.text_embed_model.embed_size

        self.root = root

        self.id_to_word = {}
        self.word_to_id = {}
        filename = os.path.join(self.root, "Sentiments_List.txt")
        # fill up dictionary from Sentiments_List.txt file
        f = open(filename, "r", encoding="latin-1")
        lines = f.readlines()
        for line in lines:
            sentiment = re.search("""(?<=ABBREVIATION: ").+(?=")""", line)
            index = re.search("""\d+(?=.)""", line)
            start_s, end_s = sentiment.span()
            start_i, end_i = index.span()

            # index
            id = int(line[start_i:end_i])
            # sentiment
            word = line[start_s:end_s]

            # add to dictionary
            self.id_to_word[id] = word
            self.word_to_id[word] = id

    def transform(self, target_lst):
        """
        Transform the target_lst of sentiments provided by the PITTs dataset to
        a Pytorch tensor based on the Word2Vec model.

        target_list: a list of lists where each element is a number
        """
        # flatten list
        lst = [item for sublist in target_lst for item in sublist]

        # convert to int
        lst = [int(num) for num in lst]

        most_common_descriptor = self.id_to_word[max(lst, key=lst.count)]

        try:
            vec = self.model.get_vector(most_common_descriptor)
        except KeyError:
            vec = np.zeros([self.embed_size])

        return torch.from_numpy(np.array(vec))


class TopicsPreProcessor:
    """
    Class to pre-process the topics provided by the PITTs dataset.

    """

    def __init__(self, root="./data/annotations",
                 embed_model="glove-wiki-gigaword-300"):
        """
        Initialise a new Topics PreProcessor.

        :param root:        the folder that contains Topics_List.txt
        :param embed_model: the name of the word2vec model
        """
        self.text_embed_model = TextEmbedModel(embed_model)
        self.embed_model = self.text_embed_model.embed_model_name
        self.model = self.text_embed_model.model
        self.embed_size = self.text_embed_model.embed_size

        self.root = root

        self.id_to_word = {}
        self.word_to_id = {}

        filename = os.path.join(self.root, "Topics_List.txt")
        # fill up dictionary from Topics_List.txt file
        with open(filename, "rb") as f:
            contents = f.read()

        lines = contents.decode("utf-16")
        lines = lines.split("\n")

        for line in lines:

            if line == "":
                break

            try:
                topic = re.search("""(?<=ABBREVIATION: ").+(?=")""", line)
                start_s, end_s = topic.span()
                # topic
                word = line[start_s:end_s].replace("_", " ")
            except AttributeError:
                word = "Unclear"

            try:
                index = re.search("""\d+(?=\s)""", line)
                start_i, end_i = index.span()
                # index
                id = int(line[start_i:end_i])
            except AttributeError:
                id = 39

            # add to dictionary
            self.id_to_word[id] = word
            self.word_to_id[word] = id

    def transform(self, target_lst):
        """
        Transform the target_lst of topics provided by the PITTs dataset to
        a Pytorch tensor based on the Word2Vec model.

        target_list: a list of numbers or text
        """
        count = 0
        vec_lst = []
        num_lst = []

        for el in target_lst:
            try:
                x = int(el)
                num_lst.append(x)
                count += 1
            except ValueError:
                # Get the vector representation of this phrase
                vec_lst.append(self.text_embed_model.get_vector_rep(el))

        if count == 0:
            # The target list has all user text inputs so try to find the
            # most represented phrase
            cosines = [0] * len(vec_lst)
            for i in range(len(vec_lst)):
                for j in range(len(vec_lst)):
                    if i != j:
                        cosines[i] += cosine_sim(vec_lst[i], vec_lst[j])

            max_val = max(cosines)
            max_index = cosines.index(max_val)

            final = vec_lst[max_index]

        else:
            most_common_descriptor = \
                self.id_to_word[max(num_lst, key=num_lst.count)]

            final = self.text_embed_model.get_vector_rep(most_common_descriptor)

        return torch.from_numpy(np.array(final)).float()


class StrategiesPreProcessor:
    """
    Class to pre-process the strategies provided by the PITTs dataset.

    """

    def __init__(self, root="./data/annotations",
                 embed_model="glove-wiki-gigaword-300"):
        """
        Initialise a new Strategies PreProcessor.

        :param root:        the folder that contains Strateges_List.txt
        :param embed_model: the name of the word2vec model
        """
        self.text_embed_model = TextEmbedModel(embed_model)
        self.embed_model = self.text_embed_model.embed_model_name
        self.model = self.text_embed_model.model
        self.embed_size = self.text_embed_model.embed_size

        self.root = root

        self.id_to_word = {}
        self.word_to_id = {}

        filename = os.path.join(self.root, "Strategies_List.txt")
        # fill up dictionary from Strategies_List.txt file
        f = open(filename, "r", encoding="latin-1")
        lines = f.readlines()

        for line in lines:

            if line == "":
                break

            strategy = re.search("""(?<=, ).+""", line)
            start_s, end_s = strategy.span()
            # topic
            word = line[start_s:end_s]

            index = re.search("""\d+(?=,)""", line)
            start_i, end_i = index.span()
            # index
            id = int(line[start_i:end_i])

            # add to dictionary
            self.id_to_word[id] = word
            self.word_to_id[word] = id

    def transform(self, target_lst):
        """
        Transform the target_lst of topics provided by the PITTs dataset to
        a Pytorch tensor based on the sBERT model.

        target_list: a list of lists, each element may contain a number or text
        """

        # flatten list
        target_lst = [item for sublist in target_lst for item in sublist]

        count = 0
        vec_lst = []
        num_lst = []

        for el in target_lst:
            try:
                x = int(el)
                num_lst.append(x)
                count += 1
            except ValueError:
                # Get the vector representation of this phrase
                tagged_word_tokens = nltk.pos_tag()
                el = " ".join([])
                vec_lst.append(self.text_embed_model.get_vector_rep(el))

        if count == 0:
            # The target list has all user text inputs so try to find the
            # most represented phrase
            cosines = [0] * len(vec_lst)
            for i in range(len(vec_lst)):
                for j in range(len(vec_lst)):
                    if i != j:
                        cosines[i] += cosine_sim(vec_lst[i], vec_lst[j])

            max_val = max(cosines)
            max_index = cosines.index(max_val)

            final = vec_lst[max_index]

        else:
            most_common_descriptor = \
                self.id_to_word[max(num_lst, key=num_lst.count)]

            final = self.text_embed_model.get_vector_rep(most_common_descriptor)

        return torch.from_numpy(np.array(final)).float()


class SlogansPreProcessor:
    """
    Class to pre-process the slogans provided by the PITTs dataset.

    """

    def __init__(self):
        """
        Initialise a new Strategies PreProcessor.

        :param root:        the folder that contains Slogans.json
        """
        self.text_embed_model = SentenceEmbedModel()
        self.embed_model = self.text_embed_model.embed_model_name
        self.model = self.text_embed_model.model
        self.embed_size = self.text_embed_model.embed_size

    def transform(self, target_lst):
        """
        Transform the target_lst of Slogans provided by the PITTs dataset to
        a Pytorch tensor based on the sBERT model.

        target_list: a list of lists, each element may contain a number or text
        """

        # flatten list
        target_lst = [item for sublist in target_lst for item in sublist]
        vec_lst = [self.text_embed_model.get_vector_rep(el) for el in target_lst]

        # The target list has all user text inputs so try to find the
        # most represented phrase
        cosines = [0] * len(vec_lst)
        for i in range(len(vec_lst)):
            for j in range(len(vec_lst)):
                if i != j:
                    cosines[i] += cosine_sim(vec_lst[i], vec_lst[j])

        max_val = max(cosines)
        max_index = cosines.index(max_val)

        final = vec_lst[max_index]


        return torch.from_numpy(np.array(final)).float()

class QAPreProcessor:
    """
    Class to pre-process the QA-ActionReason provided by the PITTs dataset.

    """

    def __init__(self):
        """
        Initialise a new QA PreProcessor.

        :param root: the folder that contains QA_ActionReason.json
        """
        self.text_embed_model = SentenceEmbedModel()
        self.embed_model = self.text_embed_model.embed_model_name
        self.model = self.text_embed_model.model
        self.embed_size = self.text_embed_model.embed_size


    def transform(self, target_lst):
        """
        Transform the target_lst of QA_ActionReason provided by the PITTs dataset to
        a Pytorch tensor based on the sBERT model.

        target_list: a list of lists, each element may contain a number or text
        """

        # flatten list
        target_lst = [item for sublist in target_lst for item in sublist]
        vec_lst = [self.text_embed_model.get_vector_rep(
            self.text_embed_model.get_simplified(el)
            ) for el in target_lst]

        # The target list has all user text inputs so try to find the
        # most represented phrase
        cosines = [0] * len(vec_lst)
        for i in range(len(vec_lst)):
            for j in range(len(vec_lst)):
                if i != j:
                    cosines[i] += cosine_sim(vec_lst[i], vec_lst[j])

        max_val = max(cosines)
        max_index = cosines.index(max_val)

        final = vec_lst[max_index]

        return torch.from_numpy(np.array(final)).float()

class QA_Combined:
    """
    Preprocessor that combines qa with sentiment, topics, strategy
    """
    def __init__(self):
        slogan = 0
        qa = 0
        sentiment = 0
        strategy = 0
        self.text_embed_model = SentenceEmbedModel()
        self.embed_model = self.text_embed_model.embed_model_name
        self.model = self.text_embed_model.model
        self.embed_size = self.text_embed_model.embed_size
    
    def transform(self, target_lst):
        #TODO: get vector rep of slogan

        #TODO: get vector rep of strategy
        #TODO: get vector rep of topic
        #TODO: get vector rep of qa
        # use only one sentence embed model?
        #TODO: Concatenate vectors
        # return ONE vector in dimensions (AxB)
        
        pass

def load_annotation_json(filename="data/annotations/Sentiments.json"):
    """
    Load topics annotation

    :param filename:    annotation file name
    :return:    Dict:   a dictionary of annotation
    """

    descriptor = {}

    with open(filename, "r") as f:
        descriptor = json.load(f)
    return descriptor


def cosine_sim(vec1, vec2):
    """
    Return the cosine similarity between two word vectors.

    vec1: (np array)    the vector that represents one word
    vec2: (np array)    the vector that represents the othe word
    """
    cosine_similarity = \
        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return cosine_similarity

