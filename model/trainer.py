import math
import os
from tqdm import tqdm

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from transformers import BertConfig
from transformers import BertTokenizer

from . import models
from . import sampling
from . import util
from .entity import Dataset
from .evaluator import Evaluator
from .reader import JsonInputReader


class BaseTrainer:
    """ Trainer base class with common methods """
    def __init__(self, cfg, logger):
        self._args = cfg

        # Arguments
        self._tokenizer_path = self._args.get(
            'preprocessing', 'tokenizer_path')
        self._max_span_size = self._args.getint(
            'preprocessing', 'max_span_size')
        self._lowercase = self._args.getboolean(
            'preprocessing', 'lowercase')
        self._sampling_processes = self._args.getint(
            'preprocessing', 'sampling_processes')

        # self._label = self._args.get('logging', 'label')
        self._log_path = self._args.get('logging', 'log_path')
        # self._debug = self._args.getboolean('logging', 'debug')

        self._model_type = self._args.get('model', 'model_type')
        self._model_path = self._args.get('model', 'model_path')
        self._gpu = self._args.getint('model', 'gpu')
        self._cpu = self._args.getboolean('model', 'cpu')
        self._eval_batch_size = self._args.getint('model', 'eval_batch_size')
        self._size_embedding = self._args.getint('model', 'size_embedding')
        self._rel_filter_threshold = self._args.getfloat('model', 'rel_filter_threshold')
        self._prop_drop = self._args.getfloat('model', 'prop_drop')
        self._max_pairs = self._args.getint('model', 'max_pairs')
        self._freeze_transformer = self._args.getboolean(
            'model', 'freeze_transformer')
        self._no_overlapping = self._args.getboolean(
            'model', 'no_overlapping')

        self._types_path = self._args.get('input', 'types_path')

        self._logger = logger

        # CUDA devices
        self._device = torch.device(
            'cuda:' + str(self._gpu) if torch.cuda.is_available()
            and not self._cpu else 'cpu')
        self._gpu_count = torch.cuda.device_count()


class SpanTrainer(BaseTrainer):
    """ Joint entity extraction training and evaluation """
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(
            self._tokenizer_path, do_lower_case=self._lowercase)

        # Input reader
        self._reader = JsonInputReader(
            self._types_path,
            self._tokenizer,
            max_span_size=self._max_span_size)

        # Create model
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

        # path to export predictions to
        self._predictions_path = os.path.join(
            self._log_path, 'predictions_%s_epoch_%s.json')

    def eval(self, jdoc: list):
        dataset_label = 'prediction'

        self._logger.info("Model: %s" % self._model_type)

        # Read datasets
        self._reader.read({dataset_label: jdoc})
        self._log_datasets()

        # evaluate
        jpredictions = self._eval(
            self._model, self._reader.get_dataset(dataset_label))

        self._logger.info("Logged in: %s" % self._log_path)

        return jpredictions

    def _eval(
        self, model: torch.nn.Module,
        dataset: Dataset, epoch: int = 0,
        updates_epoch: int = 0, iteration: int = 0): # noqa

        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        evaluator = Evaluator(
            dataset, self._reader, self._tokenizer,
            self._rel_filter_threshold, self._no_overlapping,
            self._predictions_path, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self._eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self._sampling_processes,
            collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
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

                # evaluate batch
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

        jpredictions = evaluator.store_predictions()

        return jpredictions

    def _log_datasets(self):
        self._logger.info("Entity type count: %s" % self._reader.entity_type_count)

        self._logger.info("Entities:")
        for e in self._reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        for k, d in self._reader.datasets.items():
            self._logger.info('Document: %s' % k)
            self._logger.info("Sentences count: %s" % d.document_count)
            # self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % self._reader.context_size)
