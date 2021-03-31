import uuid
from app.cache.auth_code_cache import auth_code_cache
import random
from app.common.error_code import ErrorCode

from app.foundation import db, log
import hashlib

def verify_code(otp, mobile):
    if not otp:
        has_sent = auth_code_cache.exists_auth_code(mobile)
        if has_sent:
            raise exceptions.InvalidParamError(ErrorCode.ERROR_OTP_HAS_SENT)
        auth_code = random.randint(100000, 999999)
        content = '[xxx],您的验证码为{},此验证码五分钟内有效, 退订回T'.format(auth_code)
        from app.tasks import send_sms_task
        send_sms_task(content, mobile)
        auth_code_cache.save_auth_code(mobile,auth_code)
    else:
        exist_code = auth_code_cache.get_auth_code(mobile)
        if exist_code == otp:
            return True
        else:
            return False


