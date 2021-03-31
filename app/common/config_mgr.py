from time import time
from app.models.AuthPermiss import AuthPermiss
import hashlib
import random
from flask import current_app


DEFAULT_CONFIG = {
    "USE_TIDB" : False,
}

FLASK_DEFAULT_CONFIGS = {

}

def get_config(config_name):
    if config_name not in current_app.config:
        if config_name in DEFAULT_CONFIG:
            return DEFAULT_CONFIG[config_name]
        return None
    return current_app.config[config_name]

def get_flask_config_obj():
    class FlaskConfig(object):
        pass

    if config_name not in current_app.config:
        if config_name in DEFAULT_CONFIG:
            return DEFAULT_CONFIG[config_name]
        return None
    return current_app.config[config_name]