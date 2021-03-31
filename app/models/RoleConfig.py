# coding=utf-8

from app.foundation import db
from app.common.constants import UserRole

class RoleConfig(db.Model):
    role       = db.Column(db.SmallInteger, primary_key=True, default=UserRole.ROLE_NORMAL)
    resource   = db.Column(db.String(64), primary_key=True, default='')
    can_create = db.Column(db.Boolean, nullable=False, default=False)
    can_read   = db.Column(db.Boolean, nullable=False, default=False)
    can_update = db.Column(db.Boolean, nullable=False, default=False)
    can_delete = db.Column(db.Boolean, nullable=False, default=False)

    def __init__(self, **kwargs):
        for k, v in list(kwargs.items()):
            setattr(self, k, v)

    def get_info(self):
        return {
            'id':         self.id,
            'role':       self.role,
            'resource':   self.resource,
            'can_create': self.can_create,
            'can_read':   self.can_read,
            'can_update': self.can_update,
            'can_delete': self.can_delete,
        }

    def get_mask(self):
        mask = 0x0
        if self.can_create:
            mask |= 0x8
        if self.can_read:
            mask |= 0x4
        if self.can_update:
            mask |= 0x2
        if self.can_delete:
            mask |= 0x1
        return mask

    @classmethod
    def check_permission(cls, role, resource, crud='r'):
        perm = 0x0
        config = cls.query.get((role, resource))
        if config:
            mask = config.get_mask()
            requires = list(crud)
            if 'c' in requires:
                perm |= 0x8
            if 'r' in requires:
                perm |= 0x4
            if 'u' in requires:
                perm |= 0x2
            if 'd' in requires:
                perm |= 0x1
        return (perm != 0x0) and (mask & perm == perm)
