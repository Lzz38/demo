import os
import logging
import datetime

WTF_CSRF_ENABLED = False
PERMANENT_SESSION_LIFETIME = datetime.timedelta(hours=18)
SECRET_KEY = 'dsflayouyoucheckyou;you-wi-guessdfas'

JSON_SORT_KEYS = False
JSONIFY_PRETTYPRINT_REGULAR = False
BASEDIR = os.path.abspath(os.path.dirname(__file__))
SAVE_FILE_PATH = BASEDIR + '/app/static/upload/template/'
SQLALCHEMY_POOL_RECYCLE = 10
# http://stackoverflow.com/questions/33738467/how-do-i-know-if-i-can-disable-sqlalchemy-track-modifications
SQLALCHEMY_TRACK_MODIFICATIONS = False

MYSQL_USER = 'root'
MYSQL_PASS = 'root'
MYSQL_HOST = '127.0.0.1'
MYSQL_PORT = '3306'
MYSQL_DB = 'demo'

SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8mb4' % (MYSQL_USER, MYSQL_PASS, MYSQL_HOST, MYSQL_PORT, MYSQL_DB)

PASSWORD_SECRET = 'dsayouyoucheckyoudf;a'

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 1
REDIS_PWD = "root"
# Used by rqworker => current version does not recognize `REDIS_DB` :(
REDIS_URL = "redis://:%s@%s:%d/%d" % (REDIS_PWD, REDIS_HOST, REDIS_PORT, REDIS_DB)

UPLOAD_DIR = BASEDIR + '/app/static/upload/'
UPLOAD_URI = '/static/upload'

VERIFY_CODE_FONTS = [
    BASEDIR + '/app/resource/fonts/aller-bold.ttf'
]

STORAGE_ENGINE = 'local'
STORAGE_APPID = 'storage'

STORAGE_HOST = ''
STORAGE_PORT = 9090

USER_IMPORT_DIR = BASEDIR + '/app/static/import/'
SAVE_IMAGE_PATH = BASEDIR + '/app/static/upload/'

EXPORT_URI = "/export/"

BACKUP_DIR = "/backup/"
TEMPLATE_URI = "/template/"
USER_IMAGE_URI = "/img/"
WECHAT_URI = "/wechat/"
LOG_LEVEL = logging.INFO


INTER_LOCAL_URL = "http://127.0.0.1"

# token expire
ACCESS_TOKEN_EXPIRES = 7200

INTERNAL_SERVICE_TOKEN = "dsfadfadsfasdf"

SEND_EXCEPTION_MSG = False
CELERY_BROKERS_DB = 6
CELERY_BACKEND_DB = 7
# 取消被叫7天内无法重复添加限制
# USE_NEXT_DIAL_TIME = False
LIMIT_NEXT_DIAL_TIME = 7
OFFLINE_TASK_PACK = 2000
OFFLINE_TASK_SLEEP = 0
INNER_TOKEN = "DFdfa9-23$5#fsd(0=="
#CELERY_BROKERS = 'amqp://guest:guest@127.0.0.1/telemarket'
CELERY_BROKERS = "redis://:%s@%s:%d/%d" % (REDIS_PWD, REDIS_HOST, REDIS_PORT, CELERY_BROKERS_DB)
CELERY_BACKEND = "redis://:%s@%s:%d/%d" % (REDIS_PWD, REDIS_HOST, REDIS_PORT, CELERY_BACKEND_DB)

LIMIT_START_TIME = 0
LIMIT_END_TIME = 24