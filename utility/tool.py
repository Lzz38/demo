import time

from functools import update_wrapper

def cost_time_handler(clog):
    def decorator(func):
        def wrapped_function(*args,**kwargs):
            import random
            random_num = random.randint(1, 10000000)
            start_time = time.time()
            clog.info(func.__name__ + " id: " + str(random_num) + " begin to execute")
            result = func(*args,**kwargs)
            clog.info(func.__name__ + " id: " + str(random_num) + " function cost " + str(time.time() - start_time) + " second")
            return result
        return update_wrapper(wrapped_function, func)
    return decorator

def get_cur_ts(need_offset= False):
    if not need_offset:
        return int(time.time())
    return int(time.time()-time.timezone)

def get_today_half_hour():
    today_second = (get_cur_ts(need_offset=True) % (24 * 60 * 60))
    today_half_hour = today_second // (30 * 60)
    return today_half_hour