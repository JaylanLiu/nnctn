#!/usr/bin/env python
# coding: utf-8
import sqlite3
import re
from math import tanh
import math
import jieba
from collections import defaultdict
import argparse
import pandas as pd
import numpy as np
import sys
import logging

# #辅助数据及函数
ignorewords = set([
    '恶性', '恶性肿瘤', '肿瘤', '性', '/', '的', '间', '型', '状', '样', '癌', '瘤', '腺癌', ',',
    '，', '.', '。', ' ', '组织', '后', '小',
])
essentialwords = set(['癌', '瘤', '白血病'])

def dtanh(y):
    return 1.0 - y * y

#中文分词，限定中文字符
def tokenization(paragraph: str,
                 stop_words: set = ignorewords,
                 essentialwords: set = essentialwords):
    paragraph = str(paragraph)

    #如无必须含有的词，直接返回空集合
    if len(re.findall('|'.join(essentialwords), paragraph)) == 0:
        return set([])

    #只保留中文字符
    paragraph = re.sub('[^\u4e00-\u9fa5]+', '', paragraph)
    return list(set(jieba.cut_for_search(paragraph)) - stop_words)


# # 索引类
class indexer:
    # 初始化crawler类并传入数据库名称
    def __init__(self, dbname: str):
        self.con = sqlite3.connect(dbname)

    def __del__(self):
        self.con.close()

    def dbcommit(self):
        self.con.commit()

    # 辅助函数，用以获取条目的id，如果条目不存在，就将其加入数据库中
    def getentryid(self,
                   table: str,
                   field: str,
                   value: str,
                   createnew: bool = True):
        cur = self.con.execute("select rowid from %s where %s='%s'" %
                               (table, field, value))
        res = cur.fetchone()
        if res == None:
            cur = self.con.execute("insert into %s (%s) values ('%s')" %
                                   (table, field, value))
            return cur.lastrowid
        else:
            return res[0]

    # 为每个名称建立索引
    def addtoindex(self, url: str, soup: str):
        if self.isindexed(url): return
        print('Indexing {}'.format(url))

        words = tokenization(soup)

        urlid = self.getentryid('urllist', 'url', url)

        for i in range(len(words)):
            word = words[i]
            wordid = self.getentryid('wordlist', 'word', word)
            self.con.execute(
                'insert into wordlocation(urlid,wordid,location) values (%d,%d,%d)'
                % (urlid, wordid, i))

    # 如果url已经建立索引，则返回true
    def isindexed(self, url: str):
        u = self.con.execute("select rowid from urllist where url='%s'" %
                             url).fetchone()
        if u != None:
            v = self.con.execute("select * from wordlocation where urlid=%d" %
                                 u[0]).fetchone()
            if v != None:
                return True

        return False

    #从癌种规范命名的列表建立索引
    def crawl(self, lib_df: pd.DataFrame):
        lib_df['中文'].map(lambda x: self.addtoindex(x, x))
        self.dbcommit()

    #创建数据库表
    def createindextables(self):
        self.con.execute('create table urllist(url)')
        self.con.execute('create table wordlist(word)')
        self.con.execute('create table wordlocation(urlid,wordid,location)')
        self.con.execute(
            'create table link(fromid integer,toid integer)')  #保存溯源关系
        self.con.execute('create index wordidx on wordlist(word) ')
        self.con.execute('create index urlidx on urllist(url)')
        self.con.execute('create index wordurlidx on wordlocation(wordid)')

        self.dbcommit()


# # 查询类

class searcher:
    def __init__(self, dbname):
        self.con = sqlite3.connect(dbname)

    def __del__(self):
        self.con.close()

    def getmatchrows(self, q: str):
        wordids = []
        rows = []

        #分词
        words = tokenization(q)

        #这里需要限定必须是已经索引过的词
        curatedwords = [
            x[0] for x in list(self.con.execute('select word from wordlist'))
        ]
        words = [word for word in words if word in curatedwords]

        #如果没剩下要搜索的分词，直接返回空值
        if (len(words) == 0): return None

        for word in words:
            wordrow = self.con.execute(
                "select rowid from wordlist where word='%s'" %
                word).fetchone()
            wordid = wordrow[0]
            wordids.append(wordid)

            cur = self.con.execute(
                "select urlid, wordid from wordlocation where wordid=%s" %
                wordid)
            rows.extend([row for row in cur
                         ])  #rows的结构，返回一个元组组成的数据，每个元组包含两个元素，第一个为urlid，第二个为词id

        return rows, wordids

    def getscoredlist(self, rows: list, wordids: list):  #用于合并不同的评价函数输出
        totalscores = dict([(row[0], 0)
                            for row in rows])  #初始化键为urlid的，值为初始值0的dict

        #评价函数接口
        weights = [
            (1.0, self.frequencyscore(rows)),
            (1.0, self.dfidfscore(rows)),
            (2.0, self.nnscore(rows, wordids)),  #神经网络
        ]

        for (weight, scores) in weights:
            for url in totalscores:
                totalscores[url] += weight * scores[url]  #scores 也是dict结构

        return totalscores

    def geturlname(self, id: int):
        return self.con.execute("select url from urllist where rowid=%d" %
                                id).fetchone()[0]

    def query(self, q: str):
        #处理匹配字串为空的情形
        if self.getmatchrows(q) == None:
            #print('%f\t%s' % (0.0, '未知'))
            return None

        rows, wordids = self.getmatchrows(q)
        scores = self.getscoredlist(rows, wordids)
        rankedscores = sorted([(score, url)
                               for (url, score) in scores.items()],
                              reverse=1)

        #for (score, urlid) in rankedscores:#这里是否有必要输出
        #    print('%f\t%s' % (score, self.geturlname(urlid)))#输出的是结果，返回的是中间值

        return wordids, [r[1] for r in rankedscores
                         ], [r[0] for r in rankedscores]  #返回训练nn的接口

    # 归一化函数
    def normalizescores(self, scores: dict, smallIsBetter: int = 0):
        vsmall = 0.00001  #避免被0整除
        if smallIsBetter:
            minscore = min(scores.values())
            return dict([(u, float(minscore) / max(vsmall, l))
                         for (u, l) in scores.items()])
        else:
            maxscore = max(scores.values())
            if maxscore == 0: maxscore = vsmall
            return dict([(u, float(c) / maxscore)
                         for (u, c) in scores.items()])

    #单词频度评价
    def frequencyscore(self, rows: list):
        counts = dict([(row[0], 0) for row in rows])
        for row in rows:
            counts[row[0]] += 1
        return self.normalizescores(counts)

    #df*idf model
    def dfidfscore(self, rows: list):
        scores = defaultdict(float)
        for urlid, wordid in rows:
            tf = defaultdict(int)
            df = len(
                list(
                    self.con.execute(
                        "select urlid, wordid from wordlocation where wordid=%s"
                        % wordid)))  #词指向了多少url
            N = len(list(self.con.execute("select * from urllist")))  #共有多少url

            if df > 0:
                idf = math.log(N / df + 1)

                for doc_id, _ in self.con.execute(
                        "select urlid, wordid from wordlocation where wordid=%s"
                        % wordid):
                    tf[doc_id] += 1

                for doc_id in tf:
                    doc_tokens_len = len(
                        list(
                            self.con.execute(
                                "select urlid, wordid from wordlocation where urlid=%s"
                                % doc_id)))  #url包含多少分词
                    #print(word,doc_id,doc_tokens_len)
                    scores[doc_id] += tf[
                        doc_id] / doc_tokens_len * idf  #consideration of doc length

        return self.normalizescores(scores)

    #神经网络的评价体系 net的实例化发生在哪个阶段合适
    def nnscore(self, rows: list, wordids: list):
        net = searchnet('nn.db')
        urlids = [urlid for urlid in set([row[0] for row in rows])]
        nnres = net.getresult(wordids, urlids)
        scores = dict([(urlids[i], nnres[i]) for i in range(len(urlids))])
        return self.normalizescores(scores)


# nn类

class searchnet:
    def __init__(self, dbname):
        self.con = sqlite3.connect(dbname)

    def __del__(self):
        self.con.close()

    def maketables(self):
        self.con.execute('create table hiddennode (create_key)')
        self.con.execute('create table wordhidden(fromid,toid,strength)')
        self.con.execute('create table hiddenurl(fromid,toid,strength)')
        self.con.commit()

    #获取当前连接强度，若连接不存在，则返回默认值
    def getstrength(self, fromid, toid, layer):
        table = 'wordhidden' if layer == 0 else 'hiddenurl'
        res = self.con.execute(
            'select strength from %s where fromid=%d and toid=%d' %
            (table, fromid, toid)).fetchone()
        if res == None:
            if layer == 0: return -0.2
            if layer == 1: return 0
        else:
            return res[0]

    #更新连接,不存在时创建
    def setstrength(self, fromid, toid, layer, strength):
        table = 'wordhidden' if layer == 0 else 'hiddenurl'
        #print('select rowid from %s where fromid=%d and toid=%d' %(table,fromid,toid))
        res = self.con.execute(
            'select rowid from %s where fromid=%d and toid=%d' %
            (table, fromid, toid)).fetchone()
        if res == None:
            self.con.execute(
                'insert into %s (fromid,toid,strength) values (%d,%d,%f)' %
                (table, fromid, toid, strength))
        else:
            rowid = res[0]
            self.con.execute('update %s set strength=%f where rowid=%d' %
                             (table, strength, rowid))

    #对传入的单词组合，在隐藏层建立一个新节点。并初始化连接
    def generatehiddennode(self, wordids, urls):
        if len(wordids) > 3: return None
        createkey = '_'.join(sorted(str(wi) for wi in wordids))  #词的id排序后连接
        res = self.con.execute(
            "select rowid from hiddennode where create_key='%s'" %
            createkey).fetchone()

        if res == None:
            cur = self.con.execute(
                "insert into hiddennode (create_key) values ('%s')" %
                createkey)
            hiddenid = cur.lastrowid
            #设置权重
            for wordid in wordids:
                self.setstrength(wordid, hiddenid, 0, 1.0 / len(wordids))
            for urlid in urls:
                self.setstrength(hiddenid, urlid, 1, 0.1)
            self.con.commit()

    #针对输入的词和网址，返回所有的中间层id
    def getallhiddenids(self, wordids, urlids):
        ll = {}
        for wordid in wordids:
            cur = self.con.execute(
                'select toid from wordhidden where fromid=%d' % wordid)
            for row in cur:
                ll[row[0]] = 1

        for urlid in urlids:
            cur = self.con.execute(
                'select fromid from hiddenurl where toid=%d' % urlid)
            for row in cur:
                ll[row[0]] = 1

        return list(ll.keys())

    #利用数据库中保存的信息，建立起网络
    def setupnetwork(self, wordids, urlids):
        #值列表
        self.wordids = wordids
        self.hiddenids = self.getallhiddenids(wordids, urlids)
        self.urlids = urlids

        #节点输出
        self.ai = [1.0] * len(self.wordids)
        self.ah = [1.0] * len(self.hiddenids)
        self.ao = [1.0] * len(self.urlids)

        #建立权重矩阵
        self.wi = [[
            self.getstrength(wordid, hiddenid, 0)
            for hiddenid in self.hiddenids
        ] for wordid in self.wordids]
        self.wo = [[
            self.getstrength(hiddenid, urlid, 1) for urlid in self.urlids
        ] for hiddenid in self.hiddenids]

    #前馈法，接受一列输入，将其推入网络，返回所有输出层节点的输出结果
    def feedforward(self):
        # 查询单词是仅有的输入
        for i in range(len(self.wordids)):
            self.ai[i] = 1.0

        # 中间层的活跃程度
        for j in range(len(self.hiddenids)):
            sum = 0.0
            for i in range(len(self.wordids)):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = tanh(sum)

        # 输出层节点的活跃程度
        for k in range(len(self.urlids)):
            sum = 0.0
            for j in range(len(self.hiddenids)):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = tanh(sum)

        return self.ao[:]

    #建立神经网络，并调用feedforward取得输出
    def getresult(self, wordids, urlids):
        self.setupnetwork(wordids, urlids)
        return self.feedforward()

    #反向传播进行训练
    def backPropagate(self, targets, N=0.5):
        #计算输出层误差
        output_deltas = [0.0] * len(self.urlids)
        for k in range(len(self.urlids)):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dtanh(self.ao[k]) * error

        #计算隐藏层误差
        hidden_deltas = [0.0] * len(self.hiddenids)
        for j in range(len(self.hiddenids)):
            error = 0.0
            for k in range(len(self.urlids)):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dtanh(self.ah[j]) * error

        #更新输出权重
        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change

        #更新输入权重
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] = N * change

    def trainquery(self, wordids, urlids, selectedurl):
        self.generatehiddennode(wordids, urlids)
        self.setupnetwork(wordids, urlids)
        self.feedforward()
        targets = [0.0] * len(urlids)
        targets[urlids.index(selectedurl)] = 1.0
        self.backPropagate(targets)
        self.updatedatabase()

    #存储
    def updatedatabase(self):
        for i in range(len(self.wordids)):
            for j in range(len(self.hiddenids)):
                #print(self.wordids,'______',self.hiddenids,'_____',self.wi)
                self.setstrength(self.wordids[i], self.hiddenids[j], 0,
                                 self.wi[i][j])

        for j in range(len(self.hiddenids)):
            for k in range(len(self.urlids)):
                self.setstrength(self.hiddenids[j], self.urlids[k], 1,
                                 self.wo[j][k])

        self.con.commit()

# 子命令函数定义
def build(args):
    targetdb = args.input
    lib_df = pd.read_excel(targetdb)
    lib_df = lib_df.iloc[:, :7]
    lib_df = lib_df[~lib_df.中文.duplicated()]

    # 建立index database
    print('building index database')
    crawlern=indexer('cancertypeindex.db')
    crawlern.createindextables()
    crawlern.crawl(lib_df)
    print('build index database done')

    # 建立nn network
    print('building nn network')
    net=searchnet('nn.db')
    net.maketables()
    print('build nn network done')

def search(args):
    e=searcher('cancertypeindex.db')
    sent=args.query
    print(sent)
    print('\t%s\t%s' % ('score','target'))
    result = e.query(sent)
    if result == None:
        print('\t%f\t%s' % (0.0, '未知'))
    else:
        _, urlids, scores = result
        for score, urlid in zip(scores[:10], urlids[:10]):
            print('\t%f\t%s' % (score, e.geturlname(urlid)))
    
    print()


def train(args):
    sent=args.query
    expected=args.expected

    e=searcher('cancertypeindex.db')
    net=searchnet('nn.db')

    result = e.query(sent)
    if result == None:
        print('Error, input query string find nothing')
        return
    else:
        wordids, urlids, _ = result
        cur=e.con.execute('select rowid from urllist where url="%s"' % expected).fetchone()
        if cur==None:
            print('Error, expected output is not in the index database')
            return
        else:
            urlid=cur[0]
            if urlid not in urlids:
                print('Error, expected output is not in the input query search record')
            else:
                net.trainquery(wordids,urlids[:10],urlid)
                print('update successful')


# 接口
def interface():
    parser = argparse.ArgumentParser(prog='PROG',description='nn-based cancer type Chinese specification normalization and trace')

    subparsers=parser.add_subparsers(help='sub-command help')

    # 添加子命令 build
    build_parser=subparsers.add_parser('build',help='build index database for standardard cancer type specifications and nn network')
    build_parser.add_argument('-i','--input',default='type_of_cancer-含中文名.xlsx',help='standardard cancer type specifications list')
    build_parser.set_defaults(func=build)

    # 添加子命令 search
    search_parser=subparsers.add_parser('search',help='search the query string in the index database and return hits')
    search_parser.add_argument('-q','--query',required=True,help='cancer diagnostic descriptions as query strings')
    search_parser.set_defaults(func=search)

    # 添加子命令 train
    train_parser=subparsers.add_parser('train',help='train the nn network using user\`s query string and expected output')
    train_parser.add_argument('-q','--query',required=True,help='cancer diagnostic descriptions as query strings')
    train_parser.add_argument('-e','--expected',required=True,help='expected output')
    train_parser.set_defaults(func=train)

    parser = parser.parse_args()
    return parser


if __name__ == '__main__':
    #jieba 的输出问题需要使用logging机制解决
    logger= logging.getLogger()
    logger.propagate = False
    logger.setLevel(logging.ERROR)

    args=interface()
    args.func(args)