# coding=utf-8
from app.foundation import db, sql_log

from sqlalchemy import sql
import time

from app.common import config_mgr

def count_where(*filters):
    return db.session.query(db.func.count(1)).filter(*filters).scalar()


def exists_query(q):
    return db.session.query(q.exists()).scalar()

