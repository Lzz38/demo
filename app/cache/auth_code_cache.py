# coding=utf-8
from app.foundation import redis
from app.common.cache_base import CacheBase
import json
from app.common.json_builder import MyEncoder



class AuthCodeCache(CacheBase):

    _key = CacheBase.namespace + ':a_c_{}'

    def key(self, id):
        return self._key.format(id)

    def exists_auth_code(self, mobile):
        key = self.key(mobile)
        return redis.db.exists(key)
    
    def save_auth_code(self, mobile , code):
        key = self.key(mobile)
        self.db.setex(key, json.dumps(code), 5*60)

    def get_auth_code(self, mobile):
        key = self.key(mobile)
        code_json = self.db.get(key)
        if not code_json:
            return None
        code = json.loads(code_json)
        return code
    
auth_code_cache = AuthCodeCache()
