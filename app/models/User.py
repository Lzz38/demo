import time
from datetime import datetime
from app.foundation import db
from app.common.constants import CustomerLever, CustomerPlatformService


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(128), nullable=False)  # 搜索频率
    name = db.Column(db.String(128), nullable=False)
    wechat = db.Column(db.String(128), nullable=False, default='')
    addr = db.Column(db.String(128), nullable=False, default='')
    qq = db.Column(db.String(32), nullable=False, default='')
    mobile = db.Column(db.String(32), nullable=False, default='')
    last_name = db.Column(db.String(128), nullable=False, default='')
    is_delete = db.Column(db.Boolean, default=False)  # 是否删除
    timestamp = db.Column(db.Integer, nullable=False,
                          default=time.time)
    update_time = db.Column(db.Integer, nullable=False,
                            default=time.time, onupdate=time.time)

    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for k, v in list(kwargs.items()):
            setattr(self, k, v)

    def get_info(self):
        info = {
            'id': self.id,
            'company': self.company,
            'name': self.name,
            'wechat': self.wechat,
            'addr': self.addr,
            'qq': self.qq,
            'mobile': self.mobile,
            'last_name': self.last_name,
            'timestamp': self.timestamp,
            'update_time': self.update_time,
            'is_delete' : self.is_delete
        }

        return info
