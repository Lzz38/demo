# coding=utf-8
import json
from app.foundation import redis


class CacheType:
    INT = 1
    STRING = 2
    BOOL = 3
    ARRAY = 4
    MAP = 5


class Cache:
    PREFIX = 'dm'
    TTL = 7776000  # 3 months

    @classmethod
    def _key(cls, key):
        return '{}:{}'.format(cls.PREFIX, key)

    @classmethod
    def delete(cls, key):
        key = cls._key(key)
        redis.db.expire(key, -1)

    @classmethod
    def set(cls, key, val, ttl=None):
        key = cls._key(key)
        if ttl is None:
            ttl = cls.TTL
        redis.db.setex(key, val, ttl)

    @classmethod
    def expire(cls, key, time):
        key = cls._key(key)
        redis.db.expire(key, time)

    @classmethod
    def get(cls, key):
        key = cls._key(key)
        return redis.db.get(key)

    @classmethod
    def get_all(cls, pattern='*'):
        pattern = cls._key(pattern)
        return [{key: redis.db.get(key)} for key in redis.db.scan_iter(pattern)]

    @classmethod
    def exists(cls, key):
        key = cls._key(key)
        return redis.db.exists(key)


class AutoCache(Cache):
    KEY = ""
    TYPE = CacheType.STRING

    @classmethod
    def _filter(cls, val):
        if val is None:
            return None
        elif cls.TYPE == CacheType.INT:
            val = int(val)
        elif cls.TYPE == CacheType.BOOL:
            val = False if val == "False" else True
        elif cls.TYPE == CacheType.ARRAY or cls.TYPE == CacheType.MAP:
            try:
                val = json.loads(val)
            except ValueError:
                val = None
        return val

    @classmethod
    def delete(cls):
        return Cache.delete(cls.KEY)

    @classmethod
    def set(cls, val):
        if cls.TYPE == CacheType.ARRAY or cls.TYPE == CacheType.MAP:
            val = json.dumps(val)
        Cache.set(cls.KEY, val)

    @classmethod
    def get(cls):
        if not Cache.exists(cls.KEY):
            r = cls.build_content()
            cls.set(r)

        r = Cache.get(cls.KEY)
        r = cls._filter(r)
        return r

    @classmethod
    def build_content(cls):
        pass

