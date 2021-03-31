from app.foundation import log, db
from app.models.Customer import Customer


def get_customer_conditions(name=None, company=None, wechat=None, addr=None, qq=None, contact=None, 
                            level=None, platform=None, timestamp=None, timestamp_end=None, 
                            update_time=None, update_time_end=None):
    conditions = [Customer.is_delete == False]
    if name:
        conditions.append(Customer.name.like('%' + name + '%'))
    if company:
        conditions.append(Customer.company.like('%' + company + '%'))
    if wechat:
        conditions.append(Customer.wechat == wechat)
    if addr:
        conditions.append(Customer.addr == addr)
    if qq:
        conditions.append(Customer.qq == qq)
    if contact:
        conditions.append(Customer.contact == contact)
    if level:
        conditions.append(Customer.level == level)
    if platform is not None:
        conditions.append(Customer.platform == platform)

    if timestamp is not None and timestamp_end is not None:
        if timestamp >= timestamp_end:
            raise exceptions.InvalidParamError(ErrorCode.ERROR_TIME_ERROR)
    if timestamp is not None:
        conditions.append(Customer.timestamp >= timestamp)
    if timestamp_end is not None:
        conditions.append(Customer.timestamp <= timestamp_end)


    if update_time is not None and update_time_end is not None:
        if update_time >= update_time_end:
            raise exceptions.InvalidParamError(ErrorCode.ERROR_TIME_ERROR)
    if update_time is not None:
        conditions.append(Customer.update_time >= update_time)
    if update_time_end is not None:
        conditions.append(Customer.timestamp <= update_time_end)
    return conditions


def get_order_by(order):
    order_conditions = []
    if order in [0,1,2,3,4,5,6,7,8,9]:
        if order == 0:
            order_conditions.append(Customer.name)
        elif order == 1:
            order_conditions.append(Customer.company)
        elif order == 2:
            order_conditions.append(Customer.wechat)
        elif order == 3:
            order_conditions.append(Customer.addr)
        elif order == 4: # 开放时间
            order_conditions.append(Customer.qq) 
        elif order == 5: # 计划时间
            order_conditions.append(Customer.contact)
        elif order == 6: # 添加时间
            order_conditions.append(Customer.level)
        elif order == 7: # 最后拨打时间
            order_conditions.append(Customer.platform)
        elif order == 8: # 语音模板
            order_conditions.append(Customer.timestamp)
        elif order == 9:
            order_conditions.append(Customer.update_time)
    if not order_conditions:
        order_conditions.append(Customer.id)

    return order_conditions
