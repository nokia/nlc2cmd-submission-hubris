# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

import flask
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os
import transformers

from src.onnx import prepare_onnx_generation, onnx_predict

from webapp import app
from webapp import tellina

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


model_path = 'output/nl2bash/08-21_18:08:02'
device = 'cuda'

onnx_model, lm_head, tokenizer = prepare_onnx_generation(model_path, device)


class SearchForm(FlaskForm):
    searchbox = StringField('Query:', validators=[DataRequired()])
    submit = SubmitField('Convert')


@app.route('/', methods=['GET', 'POST'])
def main():
    form = SearchForm()
    conf = "?"
    if flask.request.method == 'POST':
        query = form.searchbox.data
        results = ["", ""]
        results[0], conf = onnx_predict(onnx_model, lm_head, tokenizer, query, device)
        conf = "%.01f"%(100*conf)
        print("CONF", conf)
        try:
        	results[1] = tellina.scrape(query)
        except:
        	results[1] = "ERROR: no translation returned"
    else:
        results = ''
    return flask.render_template('search.html', title='Home', form=form, results=results)