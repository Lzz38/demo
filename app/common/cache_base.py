# coding=utf-8
from app.foundation import redis
from contextlib import contextmanager


class CacheBase(object):

    namespace = 'tm'
    _db = None
    _start_pipeline = False
    _employee = []

    @property
    def db(self):
        if self._db is None:
            self._db = redis.db
        return self._db

    @db.setter
    def db(self, db):
        self._db = db

    @contextmanager
    def bundle_pipeline(self, *args):
        try:
            self.__pipeline()
            for employee in args:
                self.__bundle(employee)
            yield self
        finally:
            self.__release_bundle()
            self.__release_pipeline()

    def __bundle(self, employee):
        employee.db = self.db
        self._employee.append(employee)

    def __release_bundle(self):
        self._db = redis.db
        for e in self._employee:
            e.db = redis.db
        self._employee = []

    @contextmanager
    def pipeline(self):
        try:
            self.__pipeline()
            yield self
        finally:
            self.__release_pipeline()

    def __pipeline(self):
        if self._start_pipeline:
            return
        self._start_pipeline = True
        self._db = redis.db.pipeline()

    def __release_pipeline(self):
        if not self._start_pipeline:
            return
        self._start_pipeline = False
        self._db = redis.db

    def execute(self):
        if self._start_pipeline:
            return self._db.execute()