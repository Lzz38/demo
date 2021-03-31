# coding=utf-8
from functools import update_wrapper
from flask import g, abort

from app.common.error_code import ErrorCode
from app.common.json_builder import error_result
from app.models.RoleConfig import RoleConfig
from flask_login import login_required


def access_control(name, crud):
    def decorator(func):
        @login_required
        def wrapped_function(*args, **kwargs):
            if not RoleConfig.check_permission(g.user.role, name, crud):
                return error_result(http_code=500, error=ErrorCode.ERROR_GRANT_NOALLOW)
            return func(*args, **kwargs)

        return update_wrapper(wrapped_function, func)

    return decorator
