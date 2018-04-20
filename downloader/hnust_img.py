# -*- coding:utf-8 -*-
import requests
import sys
import os
from PIL import Image
import PIL.ImageOps


def get_varify_code(path):
    """获取验证码"""
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(0, 250):
        varify_resp = session.get(varify_code_url, headers=headers)
        with open(os.path.join(path, "{:0>3}.png".format(i)), 'wb') as f_obj:  # 写入图片
            f_obj.write(varify_resp.content)


def initTable(threshold=120):
    """降噪处理"""
    return [1 if threshold < i else 0 for i in range(256)]


def img_convert(path):
    """图片转换底色"""
    def _img_convert(_path):
        img = Image.open(_path)
        img = img.convert('L')  # 转换为单色
        binary_image = img.point(initTable(), '1')
        img1 = PIL.ImageOps.invert(binary_image.convert('L'))  # 颜色反转
        img1.save(_path)

    # 遍历文件夹
    for dir_path, dirs, _path in os.walk(path):
        for f in _path:
            if f.endswith((".png", ".jpg", ".jpeg")):
                _img_convert(os.path.join(dir_path, f))


if __name__ == '__main__':
    session = requests.session()
    varify_code_url = 'http://kdjw.hnust.cn/kdjw/verifycode.servlet'
    # 请求头设置
    user_agent = 'User-Agent'
    user_agent_msg = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 \
    Safari/537.36'
    headers = {
        user_agent: user_agent_msg,
        'Host': 'kdjw.hnust.cn',
    }
    get_varify_code("../datasets1")
    img_convert("../datasets1")
