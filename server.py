from configparser import ConfigParser
import datetime
from flask import Flask, jsonify, request
import hashlib
import json
import nltk
import os
from pathlib import Path
from waitress import serve

from model.logger import Logger
from model.trainer import SpanTrainer

trainer = None
logger = None

app = Flask(__name__)
app.config.from_object('configure')


def start():
    global trainer
    global logger
    cfg = ConfigParser()
    configuration_path = Path(__file__).resolve(
        strict=True).parent / 'configs' / 'extract_eval.conf'
    cfg.read(configuration_path)
    logger = Logger(cfg)
    logger.info(f'Configuration parsed: {cfg.sections()}')
    trainer = SpanTrainer(cfg, logger)


@app.route('/')
def hello():
    return 'Hammer is God'


@app.route('/extract', methods=['POST'])
def extract():
    if request.method == 'POST':
        source = request.form['source']
        isForce = request.form['isForce'] == 'true'

        hl = hashlib.md5()
        hl.update(source.encode(encoding='utf-8'))
        docHash = hl.hexdigest()

        cacheDir = './data/cache'

        if not os.path.exists(cacheDir):
            os.makedirs(cacheDir)

        cacheFile = cacheDir + f'/{docHash}.json'

        if not os.path.exists(cacheFile) or isForce:
            logger.info('Try to extract new document')

            # Tokenize document
            sentenceList = nltk.sent_tokenize(source)
            tokenList = [
                nltk.word_tokenize(sentenceItem)
                for sentenceItem in sentenceList
            ]

            # Constrcut document
            document = []
            for tokenItem in tokenList:
                doc = {"tokens": tokenItem, "entities": [], "relations": []}
                document.append(doc)
            logger.info(f'{len(document)} sentences constructed')

            # Predict sentences
            startTime = datetime.datetime.now()
            result = trainer.eval(jdoc=document)
            endTime = datetime.datetime.now()
            logger.info(
                f'Predicting time: {(endTime - startTime).microseconds} μs')
            logger.info(f'{len(result)} results predicted')

            # Cache sentences
            with open(cacheFile, "w") as f:
                json.dump({'result': result}, f)
                logger.info(f'Result {docHash} cached')
        else:
            logger.info('Try to fetch cached result')

            # Cache sentences
            with open(cacheFile, 'r') as f:
                loaded = json.load(f)
                result = loaded['result']
                logger.info(f'Cached {docHash} fetched')

        return jsonify({'result': result})
    return jsonify({'result': []})


if __name__ == "__main__":
    start()
    # app.run(host='0.0.0.0', port=5000, debug=False)
    serve(app, host="0.0.0.0", port=5000)
