import functools
import json
import requests
import eventlet
from celery import Celery, platforms
from celery.schedules import crontab
from config import CELERY_BACKEND, CELERY_BROKERS, BACKUP_DIR
from datetime import timedelta
from app.foundation import log, set_task_config, db, task_log
from functools import update_wrapper
import time
from app import app
from app.logic import import_logic,add_dialtask_logic
from utility.tool import cost_time_handler
from app.logic.email_logic import send_registered_email, send_exception_email
import os
from app.logic.offline_logic import exec_offline_task
from app.logic.mobile_to_private_logic import private_to_public,public_to_private
from app.logic.call_task_logic import get_invalid_task
from config import SEND_EXCEPTION_MSG
from kombu import Queue, Exchange
from kombu.common import Broadcast
from celery import signature

eventlet.monkey_patch(select=False)
platforms.C_FORCE_ROOT = True

task_log.info("task loading...............")
def make_celery(app):
    celery = Celery(app.import_name, broker=CELERY_BROKERS, backend=CELERY_BACKEND)
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                set_task_config()
                result = None
                try:
                    result = TaskBase.__call__(self, *args, **kwargs)
                    db.session.close()
                except Exception as e:
                    db.session.rollback()
                    db.session.close()
                    log.exception(e)
                    if SEND_EXCEPTION_MSG:
                        import sys, traceback
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        e_text = repr(traceback.format_exception(exc_type, exc_value,
                                                exc_traceback))
                        e_text = e_text.replace('\\n\'', '<br/>')
                        send_except_email(e_text)
                return result
    celery.Task = ContextTask
    return celery

celery = make_celery(app)

celery.conf.update(
    timezone = 'Asia/Shanghai',
    beat_schedule= {
        # 'add-every-10-seconds': {
        #     'task': 'app.tasks.test',
        #     'schedule': timedelta(seconds=10)
        # },
        'add-every-1800-seconds': {
            'task': 'app.tasks.dailduring',
            'schedule': timedelta(seconds=1800)
        },

    },
    task_default_queue = 'default',
    task_queues = (
        Queue('default',    routing_key='app.tasks.#'),
        Broadcast('bcnotify_tasks',),
    ),
    task_default_exchange_type = 'direct',
    task_default_routing_key = 'tasks.default',
    task_routes = ([
        ('app.tasks.*', {'queue': 'default'}),
    ],),
    broker_transport_options = {
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.2,
    }
)



@celery.task(name='app.tasks.test')
def test():
    log.info("test suc+++++++")
    return 'test suc+++++++ '



@celery.task(name='app.tasks.send_sms')
@cost_time_handler(log)
def send_sms_task(content, mobile):
    log.info('send sms begin')
    from app.logic.send_sms_logic import send_sms
    send_sms(content, mobile)
    log.info('send sms complete')
