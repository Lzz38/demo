# coding=utf-8
import time
import pytz

TIME_ZONE_INFO = pytz.timezone('Asia/Shanghai')

class UserRole:
    ROLE_ROOT = 1
    ROLE_ADMIN = 2
    ROLE_NORMAL = 3
    ROLE_CUSTOMER = 4

    state_mapping = {
        ROLE_ROOT: 'root',
        ROLE_ADMIN: '管理员',
        ROLE_NORMAL: '普通用户',
        ROLE_CUSTOMER: '客服'
    }

    @classmethod
    def get_desc(cls, key):
        if key in cls.state_mapping:
            return cls.state_mapping.get(key)
        return 'unknown'

    @classmethod
    def get_info(cls):
        return [{"id": k, "name": v} for k, v in cls.state_mapping.items()
                if k != cls.ROLE_ROOT]


class Gender:
    UNKNOWN = 0
    MALE = 1
    FEMALE = 2

    state_mapping = {
        UNKNOWN: '未知',
        MALE: '男',
        FEMALE: '女',
    }

    @classmethod
    def get_desc(cls, key):
        if key in cls.state_mapping:
            return cls.state_mapping.get(key)
        return 'unknown'


class AccessType:
    USER_ACCESS = 1
    INTERNAL_ACCESS = 2
    API_TOKEN_ACCESS = 3

    state_mapping = {
        USER_ACCESS: '用户访问',
        INTERNAL_ACCESS: '内部服务访问',
        API_TOKEN_ACCESS: '第三方访问',
    }

    @classmethod
    def get_desc(cls, key):
        if key in cls.state_mapping:
            return cls.state_mapping.get(key)
        return 'unknown'


class CustomerLever:
    ORD_CUSTOMER = 'C'
    VIP_CUSTOMER = 'B'
    SEN_CUSTOMER = 'A'

    state_mapping = {
        ORD_CUSTOMER: '普通客户',
        VIP_CUSTOMER: '会员客户',
        SEN_CUSTOMER: '重点客户',

    }

    @classmethod
    def get_desc(cls, key):
        if key in cls.state_mapping:
            return cls.state_mapping.get(key)
        return 'unknown'


class CustomerPlatformService:
    JINGDONG_1 = 1
    TIANMAO_1 = 2
    TIANMAO_2 = 3
    PINDUODUO = 4
    SUNING = 5
    JINGDONG_2 = 6

    state_mapping = {
        JINGDONG_1: '京东专卖店',
        JINGDONG_2: '京东旗舰店',
        TIANMAO_1: '淘宝专卖店',
        TIANMAO_2: '淘宝专营店',
        PINDUODUO: '拼多多',
        SUNING: '苏宁'
    }

    @classmethod
    def get_desc(cls, key):
        if key in cls.state_mapping:
            return cls.state_mapping.get(key)
        return 'unknown'
