import json
import operator
import time
import uuid
import random
from app.common.oauth import TokenHelper
from flask_cors import cross_origin
from app.common.access import access_control

from flask import Blueprint, g, request
from app.common.json_builder import success_result
from app.foundation import db, log
from app.common import exceptions
from app.common.error_code import ErrorCode
from app.models.User import User

from app.common.view_helper import get_pagination, page_format, exception_handler, get_param

try:
    from config import EXPIRY
except ImportError:
    EXPIRY = 60 * 60 * 24
user_bp = Blueprint('user_bp', __name__, template_folder='user')

@user_bp.before_request
def pre_process():
    pass


@user_bp.route('/profile', methods=['GET'])
@exception_handler
@access_control(user_bp.name, 'r')
def get_user():
    """
    get_user
    ---
    tags:
      - user
    """
    params = request.args or request.form or request.get_json(silent=True)
    result = g.user.get_info()
    return success_result(data=result)


@user_bp.route('/token',  methods=['POST'])
@exception_handler
def login():
    """
    login
    ---
    tags:
      - user
    parameters:
      - name: mobile
        in : formData
        type: string
        required: true
        description: 用户名
      - name: otp
        in : formData
        type: string
        required: true
        description: 用户名
    """
    params = request.args or request.form or request.get_json(silent=True)
    mobile = get_param(params, 'mobile', str, False, None)
    otp = get_param(params, 'otp', str, False, None)

    from app.logic.verify_code_logic import verify_code
    is_right = verify_code(otp, mobile)
    if not is_right:
        raise exceptions.InvalidParamError(ErrorCode.ERROR_OTP_ERROR)
    user = User.query.filter(User.mobile==mobile, User.is_delete==False).first()
    if not user:
        raise exceptions.InvalidParamError(ErrorCode.ERROR_CUSTOMER_NOT_EXIST)
    token_data = TokenHelper.save_token(user.id)
    result = {}
    result["access_token"] = token_data
    result["refresh_token"] = token_data
    result['expiry'] = EXPIRY
    return success_result(result)
    

@user_bp.route('/otp', methods=['POST'])
@exception_handler
def set_otp():
    """
    set_otp
    ---
    tags:
      - user
    parameters:
      - name: mobile
        in : formData
        type: string
        required: true
        description: 用户名
    """
    params = request.args or request.form or request.get_json(silent=True)
    mobile = get_param(params, 'mobile', str, False, None)
    from app.logic.verify_code_logic import verify_code
    is_right = verify_code(None, mobile)
    return success_result('success')

@user_bp.route('/login', methods=['GET'])
@exception_handler
def get_login():
  if g.user and g.user.is_authenticated():
      return redirect("/")
  return send_file('static/index.html')