# 1、动手学深度学习笔记-线性回归
## 1.1、课后习题错题总结
		
>第一题  假如你正在实现一个全连接层，全连接层的输入形状是7×8，输出形状是7×1，其中7是批量大小，
则权重参数ww和偏置参数bb的形状分别是____和____<br>
>>答案：w：1x8 ,b：1x1
>第三题 在线性回归模型中，对于某个大小为3的批量，标签的预测值和真实值如下表所示：<br>
>y_hat  y   <br>
>2.33	3.14 <br>
>1.07	0.98 <br>
>1.23	1.32 <br>
>>答案：<br>
>>```python
>>y_hat = torch.tensor([2.33, 1.07, 1.23])
 print(y_hat)
 y = torch.tensor([3.14, 0.98, 1.32])
 a = squared_loss(y_hat,y)
 print(a.sum())