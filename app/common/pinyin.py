# coding=utf-8
import pypinyin

def get_pinyin(name):
    try:
        return pypinyin.slug(name, style=pypinyin.NORMAL, separator='')
    except:
        return ''
