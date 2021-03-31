# coding=utf-8
import time, datetime
import os
from flask import Flask, g, request, send_file, redirect, url_for, jsonify,Response

from flask_login import current_user
from werkzeug.datastructures import Headers

from app.foundation import db, login_manager, redis, csrf, log, db_mgr
from app.models.User import User
from app import views
from app.common import exceptions, error_code
from app.common.oauth import TokenHelper
from app.common.json_builder import error_result
from config import INTERNAL_SERVICE_TOKEN
#from app.common.redis_session import RedisSessionInterface
from flask.json import JSONEncoder

from app.common.constants import AccessType, UserRole
from flasgger import Swagger
import calendar
from app.common.constants import TIME_ZONE_INFO
from flask.sessions import SecureCookieSessionInterface

try:
    from config import INNER_AUTH_HEADER, INNER_AUTH_USER_ID
except ImportError:
    INNER_AUTH_HEADER, INNER_AUTH_USER_ID = None, None


try:
    from config import NO_SESSION_LOGIN
except ImportError:
    NO_SESSION_LOGIN = False

try:
    from config import USE_CONSUL_CONFIG, CONSUL_ADDR, CONSUL_PORT, CONSUL_TOKEN, CONSUL_NS
except ImportError:
    USE_CONSUL_CONFIG = False

DEFAULT_APP_NAME = 'app'
DEFAULT_STATIC_DIR = 'app/static/'

# pylint: disable=W0612

class DisableSessionInterface(SecureCookieSessionInterface):
    """Prevent creating session from API requests."""
    def should_set_cookie(self, *args, **kwargs):
        return False

    def save_session(self, *args, **kwargs):
        return

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        try:
            if isinstance(obj, datetime.datetime):
                timezone_date = TIME_ZONE_INFO.localize(obj)
                date_str = timezone_date.strftime("%A %d %B %Y %H:%M:%S %z")
                return date_str
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return list(iterable)
        return JSONEncoder.default(self, obj)


def create_app(name=DEFAULT_APP_NAME, **settings_override):
    app = Flask(name)
    app.json_encoder = CustomJSONEncoder
    Swagger(app)
    if NO_SESSION_LOGIN:
        app.session_interface = DisableSessionInterface()
    app.static_folder = os.path.abspath(DEFAULT_STATIC_DIR)
    app.config.from_object('config')
    if USE_CONSUL_CONFIG:
        # Consul
        # This extension should be the first one if enabled:
        consul = Consul(app=app,consul_host=CONSUL_ADDR, consul_port=CONSUL_PORT, max_tries=5, token=CONSUL_TOKEN)
        # Fetch the conviguration:
        consul.apply_remote_config(namespace=CONSUL_NS + "/")
    app.config.update(settings_override)
    configure_foundations(app)
    configure_blueprint(app, views.MODULES)
    configure_csrf(app)
    return app


def configure_foundations(app):
    redis.init_app(app)
    db.app = app
    db.init_app(app)

    @app.after_request
    def releaseDB(response):
        db.session.close()
        g.query_db_session.close()
        return response

    login_manager.init_app(app)

    @login_manager.unauthorized_handler
    def unauthorized_callback():
        internal_service_token = request.headers.get("internal-access-token")
        if internal_service_token:
            return jsonify({
                "message": "internal-access-token error: {}".format(internal_service_token)
            }), 401
        user_token = request.headers.get("user-access-token")
        if user_token:
            return jsonify({
                "message": "user-access-token error: {}".format(user_token)
            }), 401
        if request.method == "GET":
            return jsonify({'error':
            {
                'type': "error_need_redirt",
                'message': url_for("user_bp.get_login"),
                'fields': []
            }}), 401
        else:
            return jsonify({'error':
            {
                'type': "error_need_redirt",
                'message': url_for("frontend.get_login"),
                'fields': []
            }}), 401

        #return redirect(url_for("frontend.get_login"))

    @login_manager.request_loader
    def load_api_user(request):
        user = None
        # check internal token
        # check user token
        
        user_token = request.headers.get("Authorization")
        if user_token:
            data = TokenHelper.get(user_token)
            if data:
                TokenHelper.expand_time(user_token)
                user = User.query.filter(User.id == data["uid"], User.is_delete == False).first()
                if not user:
                    return None

        if not user:
            log.warn("load_api_user failed, token: " + str(user_token))
        return user

    @login_manager.user_loader
    def load_user(id):
        user = User.query.filter(User.id == id, User.is_delete == False).first()
        if user and user.ex_time and user.ex_time < datetime.datetime.now() and user.role != UserRole.ROLE_ADMIN:
            raise exceptions.InvalidParamError(error_code.ErrorCode.ERROR_USER_EXPIRE)
        return user

    @app.before_first_request
    def before_first_request():
        pass

    @app.before_request
    def before_request():
        g.user = current_user
        DBSession = db_mgr.get_session("slave1")
        g.query_db_session = DBSession()
        from utility.tool import get_cur_ts
        now = get_cur_ts()
        g.TIMESTAMP = now


def configure_blueprint(app, modules):
    for module in modules:
        app.register_blueprint(module)
    
    @app.errorhandler(404)
    def page_not_found(e):
        return send_file('static/index.html')


def configure_csrf(app):
    csrf.init_app(app)


app = create_app()
