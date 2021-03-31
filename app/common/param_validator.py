# coding=utf-8

import re
from app.common.error_code import ErrorCode
from app.common.exceptions import InvalidParamError

class Validator:

    def __init__(self):
        pass

    @classmethod
    def validate_name(cls, name):
        if name:
            return True
        raise InvalidParamError(ErrorCode.ERROR_PARAM_NAME_ERROR)

    @classmethod
    def validate_ip(cls, ip):
        if re.match('(^\d{1,3}(\.\d{1,3}){3}$)|(^localhost$)', str(ip)):
            return True
        raise InvalidParamError(ErrorCode.ERROR_PARAM_NAME_ERROR)
