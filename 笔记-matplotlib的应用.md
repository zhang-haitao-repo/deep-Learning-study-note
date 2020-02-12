## 用matplotlib做3D图形
1.首先导入模块
```python
from mpl_toolkits.mplot3d import Axes3D
```
2.其次创建画布并导入数据
```python
fig = plt.figure()
ax=Axes3D(fig)
ax.scatter(features[:, 0].numpy(),features[:, 1].numpy(), labels.numpy());
plt.show()
```
