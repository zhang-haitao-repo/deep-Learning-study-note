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
ax.plot(features[:, 0].numpy(),features[:, 1].numpy(), labels.numpy());
ax.bar(features[:, 0].numpy(),features[:, 1].numpy(), labels.numpy(),zdir = 'x',color=['r','green','yellow','purple']);

plt.show()

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
 
X,Y = np.meshgrid(x,y)
Z = np.sqrt(X**2+Y**2)
 
ax.plot_surface(X,Y,Z)

```
