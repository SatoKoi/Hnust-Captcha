# -*- coding:utf-8 -*-
import os
from concurrent.futures import ProcessPoolExecutor
import time
from PIL import Image
import numpy as np
import uuid
from functools import reduce, wraps


def functime(f):
    """统计函数运行时间"""
    @wraps(f)
    def wrap_it(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        lasting = time.time() - start
        print("{:.2f} s".format(lasting))
        return res
    return wrap_it


@functime
def img_split_and_save(root_path, debug=False):
    """分割图片"""

    @functime
    def _img_split(path):
        """内置分割函数"""
        pix = np.array(Image.open(path))
        end = path[:path.rindex("/")].rindex("/") + 1
        letter_path = os.path.join(path[:end], 'letters')
        if not os.path.exists(letter_path):
            os.mkdir(letter_path)
        # 分割图片
        start, step = 3, 9
        col_ranges = []
        for _ in range(4):
            col_ranges.append([start, start + step])
            start = start + step + 1

        # 多进程处理
        for res in p.map(future_map, wrap_fours(pix), col_ranges, wrap_fours(debug)):
            save_path = letter_path + '/' + str(uuid.uuid4()) + ".png"
            res.save(save_path)

    def wrap_fours(data):
        """四次包装"""
        if not isinstance(data, np.ndarray):
            return np.full((1, 4), data).reshape(-1)
        return [data for _ in range(4)]

    # 遍历文件夹
    for dir_path, dirs, _path in os.walk(root_path):
        for f in _path:
            if f.endswith((".png", ".jpg", ".jpeg")):
                _img_split(os.path.join(dir_path, f))
    return True


def future_map(pix, col_range, debug):
    """多进程运行"""
    letter = pix[1:-1, col_range[0]:col_range[1]]
    letter = filter_bit(letter, debug=debug)
    return Image.fromarray(np.uint8(letter))


def filter_bit(bit_array, threhold=255, keep=1, debug=False):
    """过滤干扰线"""

    def constant_check(point, up, down, left, right):
        """将单个不连续的有效bit位去掉"""
        nonlocal bit_array
        if up >= 0 and bit_array[up, point[1]] == threhold:
            return True
        if down <= bit_array.shape[0] - 1 and bit_array[down, point[1]] == threhold:
            return True
        if left >= 0 and bit_array[point[0], left] == threhold:
            return True
        if right <= bit_array.shape[1] - 1 and bit_array[point[0], right] == threhold:
            return True
        return False

    # TODO: 需改进效率
    def cluster_keep(points, reserve=1):
        """bit位连续集合保留"""

        def find_next(constant_node, L):
            """寻找连续集合, 每个结点只向右向下进行查找,
               向左向上会进入递归地狱。
            """
            nonlocal index
            if constant_node[1] + 1 < bit_array.shape[1] and bit_array[constant_node[0], constant_node[1] + 1] == threhold:
                right_index = [constant_node[0], constant_node[1] + 1]
                index = index - set("{}{}".format(right_index[0], right_index[1]))
                L.append(right_index)
                find_next(right_index, L)
            if constant_node[0] + 1 < bit_array.shape[0] and bit_array[constant_node[0] + 1, constant_node[1]] == threhold:
                down_index = [constant_node[0] + 1, constant_node[1]]
                index = index - set("{}{}".format(down_index[0], down_index[1]))
                L.append(down_index)
                find_next(down_index, L)
            return np.vstack((L))

        def cluster_concat(s, s1):
            """集群连接, 用于reduce中的function"""
            if not s:
                return s1
            if s & s1:
                return s.union(s1)
            else:
                return s

        assert points.shape[1] == 2, "the numbers of feature of bit array must be 2"
        constant_node = points[0]
        index = set(["{},{}".format(ver, hor) for ver, hor in points]) - set("{},{}".format(constant_node[0], constant_node[1]))  # 去重
        L = [constant_node]
        sets = []  # 收集多个连续集群

        # 递归查找相邻结点
        while len(index) != 0:
            new = find_next(constant_node, L)
            sets.append(set(["{},{}".format(ver, hor) for ver, hor in new]))
            next_node = index.pop()
            constant_node = [int(idx) for idx in next_node.split(",")]
            L = [constant_node]

        # 合并拥有公共结点的集群
        filter_sets = []
        for i in range(len(sets)):
            flag = True
            cur_set = reduce(cluster_concat, sets, sets[i])
            for s in filter_sets:
                if s == cur_set:
                    flag = False
                if len(s & cur_set) > 0:
                    cur_set = cur_set | s
            if flag:
                filter_sets.append(cur_set)
        # 根据集群长度排序
        filter_sets.sort(key=lambda s: len(s), reverse=True)
        debugger([filter_sets, len(filter_sets[:reserve][0])], sep='\t')
        return filter_sets[:reserve]

    def _convert(reserve):
        """转换成相应索引"""
        return np.array(sorted([s.split(",") for res in reserve for s in res]))

    def debugger(format_, sep='\n'):
        if debug == True:
            for string in format_:
                print(format_, end=sep)

    # 选出值为1的序列
    assert isinstance(bit_array, np.ndarray), \
        'bit_array must be a numpy array'
    debugger(bit_array)
    bit_points = np.argwhere(bit_array == threhold)

    # 单个结点去除, 效率高
    filter_array = np.array([], dtype=int).reshape(-1, 2)
    for point in bit_points:
        up, down, left, right = point[0] - 1, point[0] + 1, point[1] - 1, + point[1] + 1
        status = constant_check(point, up, down, left, right)
        if status is False:
            filter_array = np.vstack((filter_array, point))
    for array in filter_array:
        bit_array[array[0], array[1]] = 0

    # bit_points = np.argwhere(bit_array == threhold)

    # 集群保留
    # debugger([bit_points, len(bit_points)], sep='\t')
    # reserve = _convert(cluster_keep(bit_points, reserve=keep))
    # bit_array = np.full(shape=(bit_array.shape[0], bit_array.shape[1]), fill_value=0)
    # for r in reserve:
    #     bit_array[int(r[0]), int(r[1])] = threhold
    debugger(bit_array)
    return bit_array


if __name__ == '__main__':
    p = ProcessPoolExecutor(4)
    img_split_and_save('../datasets/')
