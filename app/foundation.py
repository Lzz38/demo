# coding=utf-8
import math
import logging

from sqlalchemy import func
from flask_sqlalchemy import SQLAlchemy, Pagination, BaseQuery
from flask_login import LoginManager
import redis as redis_connection
from flask_wtf.csrf import CsrfProtect
from flask import abort
from cloghandler import ConcurrentRotatingFileHandler
import logging.handlers
from werkzeug.local import LocalProxy
from flask_alchemydumps import AlchemyDumps
from config import BACKUP_DIR, BASEDIR
from flask import g
# from .common.storage import LocalStorage, HttpStorage, AliOssStorage, QiNiuStorage
from app.common.DBManager import DBManager

try:
    from config import STORAGE_APPID, STORAGE_ENGINE, STORAGE_HOST, STORAGE_PORT, STORAGE_BASE_URL
except ImportError:
    STORAGE_APPID, STORAGE_ENGINE, STORAGE_HOST, STORAGE_PORT, STORAGE_BASE_URL = \
        'default_app', 'local', '', 0, ''

try:
    from config import LOG_LEVEL, LOG_FILE, HB_LOG_FILE, TASK_LOG
except ImportError:
    LOG_LEVEL, LOG_FILE, HB_LOG_FILE, TASK_LOG, SQL_LOG = logging.INFO, './log/web/view.log', './log/web/hb-view.log', './log/web/task.log', './log/web/sql.log'

try:
    from config import ACCESS_KEY_ID, ACCESS_KEY_SECRET, BUCKET_NAME
except ImportError:
    ACCESS_KEY_ID, ACCESS_KEY_SECRET, BUCKET_NAME = None, None, None

class Redis(object):
    def __init__(self):
        self._db = None

    def init_app(self, app):
        self._db = redis_connection.Redis(host=app.config['REDIS_HOST'],
                                          port=app.config['REDIS_PORT'],
                                          db=app.config['REDIS_DB'],
                                          password=app.config['REDIS_PWD'],
                                          decode_responses=True)

    @property
    def db(self):
        return self._db

class SqlAlchemyDump(object):
    def __init__(self):
        self._alchemydumps = None

    def init_app(self, app, db):
        self._alchemydumps = AlchemyDumps(app, db, BACKUP_DIR)

    @property
    def alchemydumps(self):
        return self._alchemydumps


def paginate(self, page, per_page=20, error_out=False, known_total=None):
    if error_out and page < 1:
        abort(404)

    if known_total is None:
        total = self.session.execute(
            self.statement.with_only_columns([func.count()]).order_by(None)
        ).scalar()
    else:
        total = known_total

    if total and total > 0:
        # total = min(100000 * per_page, total)
        # page = min(int(math.ceil(float(total) / float(per_page))), page)
        items = self.limit(per_page).offset((page - 1) * per_page).all()
    else:
        page = 1
        items = []

    return Pagination(self, page, per_page, total, items)


def init_logger_config():
    logger = logging.getLogger('telemarket')
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s:%(filename)s:%(funcName)-6s %(message)s')
    if LOG_FILE:
        rotateHandler = ConcurrentRotatingFileHandler(LOG_FILE, "a", 100 * 1024 * 1024, 60)
        rotateHandler.setFormatter(formatter)
        logger.addHandler(rotateHandler)
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger = logging.getLogger('telemarket_heartbeat')
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s:%(filename)s:%(funcName)-6s %(message)s')
    if HB_LOG_FILE:
        rotateHandler = ConcurrentRotatingFileHandler(HB_LOG_FILE, "a", 200 * 1024 * 1024, 60)
        rotateHandler.setFormatter(formatter)
        logger.addHandler(rotateHandler)
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger = logging.getLogger('telemarket_task')
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s:%(filename)s:%(funcName)-6s %(message)s')
    if TASK_LOG:
        fh = logging.handlers.TimedRotatingFileHandler(TASK_LOG, "midnight", backupCount=40)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        ch = logging.StreamHandler()
        logger.addHandler(ch)

        ch.setFormatter(formatter)

    logger = logging.getLogger('telemarket_sql')
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s:%(filename)s:%(funcName)-6s %(message)s')
    if SQL_LOG:
        rotateHandler = ConcurrentRotatingFileHandler(SQL_LOG, "a", 100 * 1024 * 1024, 20)
        rotateHandler.setFormatter(formatter)
        logger.addHandler(rotateHandler)
    else:
        ch = logging.StreamHandler()
        logger.addHandler(ch)

def set_logger_config():
    logger = logging.getLogger('telemarket')
    setattr(g, 'log', logger)


def set_heartbeat_config():
    logger = logging.getLogger('telemarket_heartbeat')
    setattr(g, 'log', logger)


def set_task_config():
    logger = logging.getLogger('telemarket_task')
    setattr(g, 'log', logger)
    
def get_log():
    log = getattr(g, 'log', None)
    if not log:
        set_logger_config()
        log = getattr(g, 'log', None)
    return log

def get_sql_log():
    return logging.getLogger('telemarket_sql')

def get_task_log():
    return logging.getLogger('telemarket_task')

init_logger_config()
BaseQuery.paginate = paginate
db = SQLAlchemy()
login_manager = LoginManager()
redis = Redis()
csrf = CsrfProtect()
log = LocalProxy(get_log)
sql_log = get_sql_log()
task_log = get_task_log()
alchemydumps = SqlAlchemyDump()
db_mgr = DBManager()
