# Copyright 2020 Nokia
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

import flask

app = flask.Flask(__name__)
from webapp import routes