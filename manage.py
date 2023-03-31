# coding=utf-8

from flask_script import Server, Shell, Manager, prompt_bool

from app.foundation import db, redis, log

from sqlalchemy import text
from flask_migrate import Migrate, MigrateCommand
from app import views
from app import app
from config import SECRET_KEY
from flask_alchemydumps import AlchemyDumpsCommand
from sqlalchemy import func, Index
import datetime
import json
import os
import csv
from flask_cors import *

try:
    from config import DEFAULT_COMPANY_NAME
except ImportError:
    DEFAULT_COMPANY_NAME = u'Robot'

# app = create_app()
app.debug = True
app.secret_key = SECRET_KEY
manager = Manager(app)
CORS(app, supports_credentials=True)

manager.add_command("runserver", Server('0.0.0.0', port=8200))


def _make_context():
    return dict(db=db)


manager.add_command("shell", Shell(make_context=_make_context))



def init_role():
    for res in views.MODULES:
        db.session.add(
            RoleConfig(
                role=UserRole.ROLE_ROOT, resource=res.name,
                can_create=True, can_read=True,  can_update=True, can_delete=True
            )
        )
        db.session.add(
            RoleConfig(
                role=UserRole.ROLE_ADMIN, resource=res.name,
                can_create=True, can_read=True, can_update=True, can_delete=True
            )
        )
        db.session.add(
            RoleConfig(
                role=UserRole.ROLE_NORMAL, resource=res.name,
                can_create=True, can_read=True, can_update=True, can_delete=True
            )
        )
        db.session.add(
            RoleConfig(
                role=UserRole.ROLE_CUSTOMER, resource=res.name,
                can_create=True, can_read=True, can_update=True, can_delete=True
            )
        )
    db.session.commit()




def mock_data():
    db.session.commit()


def _creat_index():
    pass


@manager.command
def createall():
    db.create_all()
    _creat_index()
    db.session.commit()

    mock_data()

@manager.command
def create_index():
    _creat_index()






@manager.command
def dropall(pass_ask=None):
    "Drops all database tables"
    if pass_ask:
        db.drop_all()
    else:
        if prompt_bool("Are you sure ? You will lose all your data !"):
            db.drop_all()


def clear_redis():
    redis.db.flushall()

@manager.command
def delete_alembic():
    sql = text('DROP TABLE alembic_version;')
    db.engine.execute(sql)


@manager.command
@manager.option('-d', '--dbname', dest='dbname')
def create_dt_index(dbname):
    engine = None
    if dbname != "master":
        from app.foundation import db_mgr
        from sqlalchemy import create_engine
        db_uri = db_mgr.get_uri(dbname)
        engine = create_engine(db_uri)
    else:
        engine = db.engine
    if not engine:
        log.error("no engine can work on")


migrate = Migrate(app, db)

#migrate = Migrate(app, db, directory="/")
manager.add_command('alchemydumps', AlchemyDumpsCommand)
manager.add_command('db', MigrateCommand)
if __name__ == "__main__":
    manager.run()
    
    
