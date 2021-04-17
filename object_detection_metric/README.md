# 1.数据格式（Data Format）
标准的PASCAL VOC 数据格式
Standart PASCAL VOC Data Format
具体格式见[VOC网站](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2008test.tar)，亦可参考`results`和`data`目录，`data`目录下`voc.name`文件修改为自己数据集的类
The Data Format Description can be found in [PASCAL SITE](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2008test.tar), you can also refer to `results` and `data` directories, If you need to estimate your own detection results, change the content in file named `voc.name` to your own class names.

# 2.MAP计算方式（The Computation Description of MAP ）
MAP计算过程描述可参考[此文]()
The Detailed Description is shown in the [article]().
# 3.代码可直接运行(Code Can Run) 
```
python voc_map_eval.py
```