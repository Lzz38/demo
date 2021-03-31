# coding=utf-8
class ErrorCode:
    ERROR_UNKNOWN = (1000, 'error_unknown', '未知错误')
    # ERROR_METHOD_ERROR = (1001, 'error_method_error', 'method not allowed')
    ERROR_INVALID_PARAM = (1002, 'error_invalid_param', '参数错误')
    ERROR_DB_ERROR = (1003, 'error_db_error', '数据库错误')
    ERROR_NOT_ALLOWED = (1004, 'error_not_allowed', '操作不允许')
    ERROR_NETWORK_ERROR = (1005, 'error_network_error', '网络错误!')
    ERROR_CACHE_ERROR = (1006, 'error_cache_error', '缓存错误!')
    ERROR_LOCK_FAILED = (1007, 'error_lock_failed', '锁定失败!')
    ERROR_NEED_LOGIN = (1008, 'error_need_login', '未登录!')
    ERROR_NEED_REDIRT = (1009, 'error_need_redirt', '重定向')

    ERROR_CUSTOMER_EXIST = (2000, 'error_customer_exist', '客户已存在')
    ERROR_CUSTOMER_NOT_EXIST = (2001, 'error_customer_not_exist', '客户不存在')

    ERROR_OTP_ERROR = (3000, 'error_otp_exist', '验证码错误')
    ERROR_OTP_HAS_SENT = (3001, 'error_otp_has_sent', '验证码已发送, 请一分钟后重试')

if __name__ == "__main__":
    import json
    e = ErrorCode()
    attrs = dir(e)
    ret = dict()
    for attr in attrs:
        if attr.startswith('ERROR_'):
            code, name, desc = getattr(e, attr, (0, 'ok'))
            ret[code] = desc
    print(json.dumps(ret))
