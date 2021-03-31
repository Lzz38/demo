#!encoding=utf-8
import string
import time
import math
import random
import hashlib
import json
from app.common.redis_cache import Cache
from config import ACCESS_TOKEN_EXPIRES


def uniqid(prefix='', more_entropy=False):
    from utility.tool import get_cur_ts
    m = get_cur_ts()
    uniqid = '%8x%05x' % (int(math.floor(m)), int((m-math.floor(m))*1000000))
    if more_entropy:
        valid_chars = list(set(string.hexdigits.lower()))
        entropy_string = ''
        for i in range(0, 10, 1):
            entropy_string += random.choice(valid_chars)
        uniqid = uniqid + entropy_string
    uniqid = prefix + uniqid
    return uniqid


def random_md5():
    key = hashlib.md5()
    key.update(uniqid().encode())
    return key.hexdigest()


class CodeHelper:

    @classmethod
    def save(cls, code, data):
        data = json.dumps(data)
        Cache.set('code:' + code, data)

    @classmethod
    def get(cls, code):
        data = Cache.get('code:' + code)
        if data is None:
            return None
        return json.loads(data)

    @classmethod
    def release_code(cls, code):
        Cache.delete('code:' + code)


class TokenHelper:

    @classmethod
    def save(cls, token, data, expire, uid):
        data = json.dumps(data)
        if Cache.exists('token:' + token):
            return False
        Cache.set('token:' + token, data)
        Cache.expire('token:' + token, expire)
        return True

    @classmethod
    def get(cls, token):
        data = Cache.get('token:' + token)
        return json.loads(Cache.get('token:' + token)) if data is not None else None
    
    @classmethod
    def expand_time(cls, token):
        Cache.expire('token:' + token, ACCESS_TOKEN_EXPIRES)

    @classmethod
    def valid(cls, token):
        token = Cache.get('token:' + token)
        return token is not None

    @classmethod
    def get_all(cls):
        return Cache.get_all("token:*")

    @classmethod
    def delete(cls, token):
        Cache.delete("token:" + token)


    @classmethod
    def save_token(cls, uid):
        # generate token
        token = random_md5()
        token_data = {
            "access_token": token,
            "uid": uid,
            "token_type": "user",
            "expires_in": ACCESS_TOKEN_EXPIRES
        }
        # save token
        if not cls.save(token_data["access_token"], token_data, ACCESS_TOKEN_EXPIRES, uid):
            return None
        return token_data

if __name__ == "__main__":
    print(random_md5())
