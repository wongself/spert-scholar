# 信息抽取模块

## 安装部署

### 环境依赖
- nltk (tested with version 3.5)
- numpy (tested with version 1.19.2)
- scikit-learn (tested with version 0.23.2)
- torch 1.1.0+ (tested with version 1.6.0)
- tqdm (tested with version 4.50.2)
- transformers 2.2.0+ (tested with version 3.4.0)

### 环境安装
1. 位于项目根目录，输入命令`cp -r /data/wsf/nobody-likes-problem/data ./`来导入信息抽取模块运行所需的预训练模型、外部数据等必要资料。
2. 上一步仅限 V100 服务器用户操作。不过，可以手动创建如下`data`文件夹，并将其放置在项目根目录。其中，`datasets`文件夹请在[此处](http://lavis.cs.hs-rm.de/storage/spert/public/datasets/scierc/)下载，`models`文件夹请在[此处](http://lavis.cs.hs-rm.de/storage/spert/public/models/scierc/)。`log`文件夹为空文件夹，存放入信息抽取模块运行日志。

```
./data
├── datasets
│   └── scierc
│       ├── scierc_dev.json
│       ├── scierc_test.json
│       ├── scierc_train_dev.json
│       ├── scierc_train.json
│       └── scierc_types.json
├── log
└── models
    └── scierc
        ├── config.json
        ├── pytorch_model.bin
        └── vocab.txt
```

## 项目开发

### 项目调试
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/spert/server.py`来启动信息抽取模块，随后在浏览器中输入本机网址及端口`2334`，来测试模块是否启动成功。若页面出现出现`Nobody Likes Problem`，则表明模块启动成功。

### 项目维护
1. 位于项目根目录，输入命令`python ./nlp/applicaitons/spert/server.py`来启动信息抽取模块。

