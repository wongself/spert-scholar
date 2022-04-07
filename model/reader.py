import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from tqdm import tqdm
from transformers import BertTokenizer

from .entity import Dataset, EntityType, RelationType, Document


class BaseInputReader(ABC):
    def __init__(self,
                 types_path: str,
                 tokenizer: BertTokenizer,
                 neg_entity_count: int = None,
                 neg_rel_count: int = None,
                 max_span_size: int = None,
                 **kwargs):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._relation_types = OrderedDict()
        self._idx2relation_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # relations
        # add 'None' relation type
        none_relation_type = RelationType('None', 0, 'None', 'No Relation')
        self._relation_types['None'] = none_relation_type
        self._idx2relation_type[0] = none_relation_type

        # specified relation types
        for i, (key, v) in enumerate(types['relations'].items()):
            relation_type = RelationType(key, i + 1, v['short'], v['verbose'],
                                         v['symmetric'])
            self._relation_types[key] = relation_type
            self._idx2relation_type[i + 1] = relation_type

        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size

        self._datasets = dict()

        self._tokenizer = tokenizer

        self._vocabulary_size = tokenizer.vocab_size

    @abstractmethod
    def read(self, dataset_path, dataset_label):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_relation_type(self, idx) -> RelationType:
        relation = self._idx2relation_type[idx]
        return relation

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonPredictionInputReader(BaseInputReader):
    def __init__(self,
                 types_path: str,
                 tokenizer: BertTokenizer,
                 max_span_size: int = None):
        super().__init__(types_path, tokenizer, max_span_size=max_span_size)

    def read(self, documents, dataset_label):
        dataset = Dataset(dataset_label, self._relation_types,
                          self._entity_types, self._neg_entity_count,
                          self._neg_rel_count, self._max_span_size)
        self._parse_dataset(documents, dataset)
        self._datasets[dataset_label] = dataset
        return dataset

    def _parse_dataset(self, documents, dataset):
        for document in tqdm(documents,
                             desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, document, dataset) -> Document:
        if type(document) == list:
            jtokens = document
        elif type(document) == dict:
            jtokens = document['tokens']
        else:
            jtokens = []

        # parse tokens
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset,
                                                 self._tokenizer)

        # create document
        document = dataset.create_document(doc_tokens, [], [], doc_encoding)

        return document


def _parse_tokens(jtokens, dataset, tokenizer):
    doc_tokens = []

    # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
    doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]

    # parse tokens
    for i, token_phrase in enumerate(jtokens):
        token_encoding = tokenizer.encode(token_phrase,
                                          add_special_tokens=False)
        if not token_encoding:
            token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
        span_start, span_end = (len(doc_encoding),
                                len(doc_encoding) + len(token_encoding))

        token = dataset.create_token(i, span_start, span_end, token_phrase)

        doc_tokens.append(token)
        doc_encoding += token_encoding

    doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]

    return doc_tokens, doc_encoding
