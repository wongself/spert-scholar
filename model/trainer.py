import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from transformers import BertConfig
from transformers import BertTokenizer

from . import models, prediction, sampling, util
from .entity import Dataset
from .reader import BaseInputReader, JsonPredictionInputReader


class BaseTrainer:
    """ Trainer base class with common methods """
    def __init__(self, cfg, logger):
        self._args = cfg

        # Arguments
        self._gpu = self._args.getint('model', 'gpu')
        self._cpu = self._args.getboolean('model', 'cpu')

        self._model_type = self._args.get('model', 'model_type')
        self._model_path = self._args.get('model', 'model_path')
        self._tokenizer_path = self._args.get('model', 'tokenizer_path')
        self._types_path = self._args.get('model', 'types_path')

        self._eval_batch_size = self._args.getint('model', 'eval_batch_size')
        self._rel_filter_threshold = self._args.getfloat(
            'model', 'rel_filter_threshold')
        self._size_embedding = self._args.getint('model', 'size_embedding')
        self._prop_drop = self._args.getfloat('model', 'prop_drop')
        self._max_span_size = self._args.getint('model', 'max_span_size')
        self._sampling_processes = self._args.getint('model',
                                                     'sampling_processes')
        self._max_pairs = self._args.getint('model', 'max_pairs')
        self._freeze_transformer = self._args.getboolean(
            'model', 'freeze_transformer')
        self._no_overlapping = self._args.getboolean('model', 'no_overlapping')
        self._lowercase = self._args.getboolean('model', 'lowercase')

        # self._label = self._args.get('logging', 'label')
        self._log_path = self._args.get('logging', 'log_path')
        # self._debug = self._args.getboolean('logging', 'debug')

        self._logger = logger

        # CUDA devices
        self._device = torch.device('cuda:' +
                                    str(self._gpu) if torch.cuda.is_available(
                                    ) and not self._cpu else 'cpu')
        self._gpu_count = torch.cuda.device_count()


class SpanTrainer(BaseTrainer):
    """ Joint entity extraction training and evaluation """
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(
            self._tokenizer_path, do_lower_case=self._lowercase)

        # input reader
        self._reader = JsonPredictionInputReader(
            self._types_path,
            self._tokenizer,
            max_span_size=self._max_span_size)

        # load model
        model_class = models.get_model(self._model_type)

        config = BertConfig.from_pretrained(self._model_path)

        self._model = model_class.from_pretrained(
            self._model_path,
            config=config,
            # Span model parameters
            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
            entity_types=self._reader.entity_type_count,
            relation_types=self._reader.relation_type_count - 1,
            max_pairs=self._max_pairs,
            prop_drop=self._prop_drop,
            size_embedding=self._size_embedding,
            freeze_transformer=self._freeze_transformer)

        # If you want to predict Spans on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu' and self._gpu_count > 1:
        #     self._model = torch.nn.DataParallel(self._model, device_ids=[0,])
        self._model.to(self._device)

    def predict(self, docs: list):
        # read datasets
        dataset = self._reader.read(docs, 'dataset')

        result = self._predict(self._model, dataset, self._reader)

        return result

    def _predict(self, model: torch.nn.Module, dataset: Dataset,
                 input_reader: BaseInputReader):
        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset,
                                 batch_size=self._eval_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=self._sampling_processes,
                                 collate_fn=sampling.collate_fn_padding)

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(
                    encodings=batch['encodings'],
                    context_masks=batch['context_masks'],
                    entity_masks=batch['entity_masks'],
                    entity_sizes=batch['entity_sizes'],
                    entity_spans=batch['entity_spans'],
                    entity_sample_masks=batch['entity_sample_masks'])
                entity_clf, rel_clf, rels = result

                # convert predictions
                predictions = prediction.convert_predictions(
                    entity_clf, rel_clf, rels, batch,
                    self._rel_filter_threshold, input_reader)

                batch_pred_entities, batch_pred_relations = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)

        result = prediction.store_predictions(dataset.documents, pred_entities,
                                              pred_relations)

        return result
