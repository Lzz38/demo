# coding=utf-8

import json
from datetime import date, datetime
from flask import jsonify
from .error_code import ErrorCode


def success_result(data={}, page={}):
    ret = {'data': data}
    if page:
        ret['paging'] = page
    return jsonify(ret)


def error_result(http_code = 500, error=ErrorCode.ERROR_UNKNOWN, data=[], desc = ""):
    code, name, default_desc = error
    ret = {
        'error':
        {
            'type': name,
            'message': desc if desc else default_desc,
            'fields': data
        }
    }
    return jsonify(ret), http_code


class MyEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        # if isinstance(obj, datetime.datetime):
        # return int(mktime(obj.timetuple()))
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        else:
            return json.JSONEncoder.default(self, obj)
