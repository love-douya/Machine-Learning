# 单层二分类感知机模型

import copy

trainint_set = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
w = [0, 0]
b = 0

def update(item):
    global w, b
    w[0] += 1 * item[1] * item[0][0] #第一个分量更新
    w[1] += 1 * item[1] * item[0][1] #第二个分量更新
    b += 1 * item[1]
    print('w = ', w, 'b = ', b)

#返回y = yi(w * x + b)的结果
def judge(item):
    res = 0
    for i in range(len(item[0])):
        #对应公式w * x
        res += item[0][i] * w[i]
    #对应公式w * x + b
    res += b
    #对应公式yi(w * x + b)
    res *= item[1]
    return res

def check():
    flag = False
    for item in trainint_set:
        if judge(item) <= 0: #如果还有误分类点，那么小于等于0
            flag = True
            update(item)
    return flag

if __name__ == '__main__':
    flag = False
    for i in range(1000):
        if not check():
            flag = True
            break
    if flag:
        print('在1000次内分类正确')
    else:
        print('1000次内分类失败')