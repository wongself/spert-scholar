from configparser import ConfigParser
import datetime
from flask import Flask, jsonify, request
import nltk
from pathlib import Path
from waitress import serve

from model.logger import Logger
from model.trainer import SpanTrainer

trainer = None
logger = None

app = Flask(__name__)
app.config.from_object('configure')


def init_extarct():
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
def hello_world():
    return 'Nobody Likes Problem'


@app.route('/query_extarct', methods=['POST'])
def query_extarct():
    if request.method == 'POST':
        source = request.form['source']
        jsentences = nltk.sent_tokenize(source)
        jtokens = [nltk.word_tokenize(jsentence) for jsentence in jsentences]

        # Parse document
        jdocument = []
        for jtoken in jtokens:
            doc = {"tokens": jtoken, "entities": [], "relations": []}
            jdocument.append(doc)
        logger.info(f'Document parsed: {jdocument}')

        # Predict
        start_time = datetime.datetime.now()
        jextract = trainer.eval(jdoc=jdocument)
        end_time = datetime.datetime.now()
        logger.info(f'Predicting time: {(end_time - start_time).microseconds} μs')
        logger.info(f'Predicted result: {jextract}')

        return jsonify({'jextract': jextract})
    return jsonify({'jextract': ''})


if __name__ == "__main__":
    init_extarct()
    # app.run(host='0.0.0.0', port=2334, debug=False)
    serve(app, host="0.0.0.0", port=2334)
