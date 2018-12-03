# Affective

## FERPlus

### How to run

To run FERPlus,follow several steps:

Step 1:cd into FERPlus directory

Step 2:generate 2 json files

```shell
python3 get_json_DIS.py # generate info_DIS.json
python3 get_json_MV.py #generate info_MV.json
```

These two json files include all data about FERPlus,so they are very large(~8G in total) and it takes approximately 18 minutes to generate them.We mainly use info_DIS.json,but info_MV.json also helps us improve the accuracy.The function of two json files are listed below:

| json file     | usage                                     |
| ------------- | ----------------------------------------- |
| info_DIS.json | images with probability distribution tags |
| info_MV.json  | images with one-hot tags                  |

Step3:

python3 main_lr_0.0001.py   (initial learning rate=0.0001),or

python3 main_lr_0.001.py(initial learning rate=0.001)

Training tricks:

Firstly,train with info_DIS.json(use probability distribution tags),after it nearly converges,use info_MV.json to reach the summit.Although probability distribution tags describe the affection better than major voting(one hot),if you use probability tags only,the loss function will never reach bottom.

Please revise here:

```python
...
print("Data loading")
with open("info_DIS.json") as f:#revise to:with open("info_MV.json") as f:
    d = json.load(f)
shuffle()
lengthpublic = len(d["PublicTest"])
...
```

### Files for downloading

Organized datasets and two json files could be downloaded here:

url:https://pan.baidu.com/s/1adg0JLiMkDb7YMLZe71yJQ

password:dfig

or connect us: miaosi2018@sari.ac.cn,  2904661326@qq.com,  miaosi@hust.edu.cn

