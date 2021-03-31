# coding=utf-8
import os
import uuid
import urllib.parse
from io import StringIO, BytesIO
from datetime import datetime
from threading import Thread
from functools import update_wrapper

from PIL import Image
from werkzeug.utils import secure_filename
from flask import request, render_template
# from wheezy.captcha.image import captcha, background, curve, noise, smooth, text, offset, rotate, warp

from app.foundation import log
from app.common.json_builder import error_result
from app.common.exceptions import AppError, InvalidParamError
from app.common.error_code import ErrorCode
import requests
from config import UPLOAD_URI, SEND_EXCEPTION_MSG
import kombu

def get_param(params, key, typ, nullable=False, default=None, type_group=None):
    try:
        if not params and nullable:
            return default
        val = params.get(key)
        if val is None and nullable:
            return default
        elif val is None:
            raise InvalidParamError(message="参数不能为空", data=[key])
        elif val == '' and typ != str:
            if nullable:
                return default
            else:
                raise InvalidParamError(message="参数类型错误", data=[key])
        if type_group:
            if typ(val) not in type_group.state_mapping:
                raise InvalidParamError(message="参数类型错误", data=[key])
        return typ(val)
    except Exception as e:
        print(e)
        raise InvalidParamError(message="参数错误", data=[key])


def get_pagination(params, default_page=1, default_size=10):
    if params:
        current = int(params.get('page', default_page))
        size = int(params.get('size', default_size))
    else:
        current = 1
        size = 10
    return current, size


def page_format(pagination):
    if not pagination:
        return {
            'page':    1,
            'page_count':   1,
            'per_page': 1,
            'total_count': 0,
        }
    else:
        return {
            'page': pagination.page,
            'page_count': pagination.pages,
            'per_page': pagination.per_page,
            'total_count': pagination.total
        }


# str2time = lambda strtime: datetime.strptime(strtime, '%Y-%m-%d')
def get_time_period(start_param='starttime', end_param='endtime'):
    '''get param `starttime` & `endtime` from request, convert them to datetime
    object and return. If not set, return None.
    '''
    start_time = request.args.get(start_param)
    end_time = request.args.get(end_param)
    if start_time:
        start_time = datetime.strptime(start_time, '%Y-%m-%d')
    if end_time:
        end_time = datetime.strptime(end_time, '%Y-%m-%d')

    return start_time, end_time


# def rq_async(func):
# #@wraps(func)
# def _wrap(*args, **kwargs):
# q = Queue(connection=redis.db)
# return q.enqueue(func, *args, **kwargs)

# return _wrap

# def async(func):
#     def _wrap(*args, **kwargs):
#         thr = Thread(target=func, args=args, kwargs=kwargs)
#         thr.start()

#     return _wrap



def exception_handler(func):
    def wrapped_function(**kwargs):
        try:
            result = func(**kwargs)
            return result
        except AppError as e:
            log.exception(e)
            return error_result(http_code=e.http_code, error=e.code, desc=e.message, data=e.data)
        except AssertionError as e:
            log.exception(e)
            return error_result(http_code=500, error=ErrorCode.ERROR_INVALID_PARAM, desc=e.message)
        except requests.exceptions.RequestException as e:
            log.exception(e)
            return error_result(error=ErrorCode.ERROR_NETWORK_ERROR, desc="网络错误")
        # except kombu.exceptions.OperationalError as e:
        #     log.error("celery cannot connected")
        #     return error_result(error=ErrorCode.ERROR_NETWORK_ERROR, desc="内部服务网络错误，请联系我们")
        except Exception as e:
            # from app.tasks import send_except_email

            log.exception(e)
            if SEND_EXCEPTION_MSG:
                import sys, traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                e_text = repr(traceback.format_exception(exc_type, exc_value,
                                          exc_traceback))
                e_text = e_text.replace('\\n\'', '<br/>')
                # send_except_email.delay(e_text)
            return error_result(error=ErrorCode.ERROR_UNKNOWN, desc = "啊哦，探索到了未知领域")
    return update_wrapper(wrapped_function, func)