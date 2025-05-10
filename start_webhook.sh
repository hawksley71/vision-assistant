#!/bin/bash
export FLASK_APP=src.core.webhook
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000 