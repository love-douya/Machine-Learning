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
        res += item[0][i] * w[i]
    #对应公式w * x
    res += b
    #对应公式w * x + b
    