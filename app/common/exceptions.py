from app.common.error_code import ErrorCode


class AppError(RuntimeError):
    def __init__(self, code=ErrorCode.ERROR_UNKNOWN, message='',
                 http_code=400, data=[]):
        super(AppError, self).__init__()
        self.code = code
        self.http_code = http_code
        self.data = data
        self.message = message


class LockFailedError(AppError):
    def __init__(self, message=''):
        super(LockFailedError, self).__init__(
            ErrorCode.ERROR_LOCK_FAILED, message, 423)


class DBError(AppError):
    def __init__(self, message=''):
        super(DBError, self).__init__(
            ErrorCode.ERROR_DB_ERROR, message, 507)


class CacheError(AppError):
    def __init__(self, message=''):
        super(CacheError, self).__init__(
            ErrorCode.ERROR_CACHE_ERROR, message, 503)


class InvalidParamError(AppError):
    def __init__(self, code=ErrorCode.ERROR_INVALID_PARAM, message='', data=[]):
        super(InvalidParamError, self).__init__(
            code=code, message=message, data=data, http_code=400)


class PermissionError(AppError):
    def __init__(self, code=ErrorCode.ERROR_NOT_ALLOWED, message=''):
        super(PermissionError, self).__init__(
            code=code, message=message, http_code=403)


class ResourceExistError(AppError):
    def __init__(self, code, message=''):
        super(ResourceExistError, self).__init__(
            code, message, 403)


class ResourceNotExistError(AppError):
    def __init__(self, code, message=''):
        super(ResourceNotExistError, self).__init__(
            code, message, 403)


class AccessTokenError(AppError):
    def __init__(self, code=ErrorCode.ERROR_UNKNOWN, message=''):
        super(AccessTokenError, self).__init__(
            code=code, message=message, http_code=203)

class NeedRedirtError(AppError):
    def __init__(self, code=ErrorCode.ERROR_NEED_REDIRT, message=''):
        super(NeedRedirtError, self).__init__(
            code=code, message=message, http_code=401)
