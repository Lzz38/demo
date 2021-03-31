import contextlib
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from config import SQLALCHEMY_DATABASE_URI
try:
    from config import SQLALCHEMY_DATABASE_URI_S1
except ImportError:
    SQLALCHEMY_DATABASE_URI_S1 = SQLALCHEMY_DATABASE_URI

try:
    from config import SQLALCHEMY_DATABASE_URI_S2
except ImportError:
    SQLALCHEMY_DATABASE_URI_S2 = SQLALCHEMY_DATABASE_URI


from sqlalchemy.orm import Query
from flask_sqlalchemy import Pagination
from sqlalchemy import func


class MyQuery(Query):
    def paginate(self, page, per_page=20, error_out=False, known_total=None):
        if error_out and page < 1:
            return None

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


class DBManager(object):

    def __init__(self):
        self.session_map = {}
        self.db_settings = {
            'slave1': SQLALCHEMY_DATABASE_URI_S1,
            'slave2': SQLALCHEMY_DATABASE_URI_S2,
        }
        self.create_sessions()
        

    def create_sessions(self):
        for role, url in self.db_settings.items():
            self.session_map[role] = self.create_single_session(url)

    @classmethod
    def create_single_session(cls, url, scopefunc=None):
        # show VARIABLES like 'wait_timeout'
        engine = create_engine(url,  pool_size=5, pool_timeout=30, pool_recycle = 30)
        return scoped_session(
            sessionmaker(
                expire_on_commit=False,
                bind=engine,
                query_cls=MyQuery,
            ),
            scopefunc=scopefunc
        )

    def get_uri(self, name):
        return self.db_settings[name]
    
    def get_session(self, name):
        try:
            if not name:
                # 当没有提供名字时，我们默认为读请求，现在的逻辑是在当前所有的配置中随机选取一个数据库，你可以根据自己的需求来调整这里的选择逻辑
                name = random.choice(self.session_map.keys())

            return self.session_map[name]
        except KeyError:
            raise KeyError('{} not created, check your DB_SETTINGS'.format(name))
        except IndexError:
            raise IndexError('cannot get names from DB_SETTINGS')

    @contextlib.contextmanager
    def session_ctx(self, bind=None):
        DBSession = self.get_session(bind)
        session = DBSession()
        try:
            yield session
            #session.commit()
        except:
            #session.rollback()
            raise
        finally:
            #session.expunge_all()
            session.close()