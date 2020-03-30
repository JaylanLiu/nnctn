# Introduction
nnctn is a nn-based cancer type Chinese specification normalization and trace-up program.

It takes variable cancer diagnostic descriptions in Chinese, normalize them to the stardard specifications in Chinese.

It has a neural network for improve the performance based on the users feedback.

It provides a api for other evaluation scrore functions.

# Usage
nnctn has three sub-commands, each for build the index database, search the query string and train the nn model.
```
$ python nnctn.py -h
usage: PROG [-h] {build,search,train} ...

nn-based cancer type specification normalization and trace

positional arguments:
  {build,search,train}  sub-command help
    build               build index database for standardard cancer type
                        specifications and nn network
    search              search the query string in the index database and
                        return hits
    train               train the nn network using user\`s query string and
                        expected output

optional arguments:
  -h, --help            show this help message and exit
```
sub-command build takes in a excel file which have a stardard Chinese cancer specifications with the column name "中文". Default is the "type_of_cancer-含中文名.xlsx" in this directory.


The program brings a pre-build "cancertypeindex.db" and a pre-build  "nn.db". Delete this two files before you execute the build sub-command.
```
$ python nnctn.py build -h
usage: PROG build [-h] [-i INPUT]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        standardard cancer type specifications list
```
sub-command search takes in a query string which represents the dignostic information in Chinese.
```
$ python nnctn.py search -h
usage: PROG search [-h] -q QUERY

optional arguments:
  -h, --help            show this help message and exit
  -q QUERY, --query QUERY
                        cancer diagnostic descriptions as query strings
```
sub-command train takes in a query string and a expected output, output must be in the query string`s search result.

There are several essential training items in the "training.set".
```
python nnctn.py train -h
usage: PROG train [-h] -q QUERY -e EXPECTED

optional arguments:
  -h, --help            show this help message and exit
  -q QUERY, --query QUERY
                        cancer diagnostic descriptions as query strings
  -e EXPECTED, --expected EXPECTED
                        expected output
```


# Exameles
## build
```
$ python nnctn.py build
building index database
Indexing 侵袭性血管黏液瘤
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 1.287 seconds.
Prefix dict has been built succesfully.
Indexing 间变性星形细胞瘤
Indexing 激活B细胞型
Indexing 急性嗜碱性白血病
Indexing 肾上腺皮质腺瘤
Indexing 腺样囊性乳腺癌
...
```


## search
10 best hits will be printed in default.
```
$ python nnctn.py search -q "甲状腺乳头状癌，手术后"
甲状腺乳头状癌，手术后
        score   target
        1.968152	乳头状甲状腺癌
        1.750000	乳头状胃腺癌
        1.550000	子宫浆膜癌/子宫乳头状浆膜癌
        1.550000	尿路上皮乳头状瘤
        1.550000	乳腺实性乳头状癌
        1.550000	乳头状肾细胞癌
        1.550000	乳头状脑膜瘤
        1.550000	黏液乳头状室管膜瘤
        1.550000	尿道上皮乳头状瘤
        1.550000	脉络丛乳头状瘤瘤
```

## train
```
$ python nnctn.py train -q "卵巢癌" -e "卵巢癌/输卵管癌"
update successful
```