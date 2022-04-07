import datetime
import logging
import os
import sys

from . import util


class Logger:
    def __init__(self, cfg):
        # Config
        self._cfg = cfg
        self._label = self._cfg.get('logging', 'label')
        self._debug = self._cfg.getboolean('logging', 'debug')

        # Logger
        self._log_formatter = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        )
        self._logger = logging.getLogger()
        self._reset_logger()

        self._timestamp = str(datetime.datetime.now()).replace(' ', '_')
        self._log_path = os.path.join(
            self._cfg.get('logging', 'log_path'), self._label, self._timestamp)
        util.create_directory(self._log_path)

        # File & Console logging
        file_handler = logging.FileHandler(
            os.path.join(self._log_path, 'all.log'))
        file_handler.setFormatter(self._log_formatter)
        self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._log_formatter)
        self._logger.addHandler(console_handler)

        if self._debug:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.INFO)

    # Logging
    def info(self, message: str):
        self._logger.info(message)

    # Reset logger
    def _reset_logger(self):
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)

        for f in self._logger.filters[:]:
            self._logger.removeFilters(f)

    @property
    def label(self):
        return self._label

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def log_path(self):
        return self._log_path
