# Notes


## record
- 在本地进行了数据处理，主要的处理方式是将`gowalla`的数据改成`train.py`中`ml1m_rating`最后处理好的效果⬇ 
    ```python
    userId,itemId,rating
         0,     0,     1
         0,     1,     1
         0,     2,     1
         0,     3,     1
         0,     4,     1
    ```
- 数据处理过程参考`LightGCN`中的`dataloader.py`的这两行代码。`trainDict`单独写的，最后保存到`gowalla_ratings.csv`文件
    >   self.__testDict = self.__build_test()
        self.__trainDict = self.__build_train()

    然后提取出每个用户交互的一系列 items
- 注意要把`gmf_config`中 user 和 item 的数量改对，用`pandas unique`统计数量， 原本`train.py`中的统计是错误的，都少了1 **(这里卡的时间比较久)**
- `gowalla`数据集比`movie`大很多，所以跑起来也比较慢


## Operations
- `python train_gowalla`