

x=0             # 设置初始点，也可随机初始化
h = 0.001       # 步长
thresh=0.001    # 设置停止阈值，若梯度小于改值则退出

# 计算梯度
def cala_grad(x):
    return 4 * x + 1.5

while True:
    # 计算当前梯度
    grad = cala_grad(x)
    # 更新参数
    x = x + h * (-1 * grad)
    # 判断是否满足停止条件
    if grad < thresh: break

print("极小值为：{}，极小值点为：{}".format(2 * x**2 + 1.5 * x + 3, x))




