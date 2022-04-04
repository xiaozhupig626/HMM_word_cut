# 步骤
# 1 根据语料库生成含有字的标识(隐藏状态)的文件
# 2 定义HMM类，获得初始矩阵，转移矩阵，发射矩阵
# 3 定义维特比算法的方法，选择出一条最优路径，作为文本的分词结果

import os
import pickle
import numpy as np
from tqdm import tqdm


# 给单词标上隐藏状态
def make_label(word):
    text_len = len(word)
    if text_len == 1:
        return "S"
    else:
        return "B" + "M" * (text_len - 2) + "E"


# 生成隐藏状态的文件
def text_to_state(file="all_train_text.txt"):
    if os.path.exists("all_train_state.txt"):  # 已存在隐藏状态文件
        return
    all_data = open(file, 'r', encoding="utf-8").read().split('\n')  ## 读取每一行的语料库的文本
    with open("all_train_state.txt", 'w', encoding='utf-8') as f:
        for d_index, data in tqdm(enumerate(all_data)):  # 遍历每一行，以空格分隔，给对应的字标上隐藏状态
            if data:
                state_ = ""
                for word in data.split(" "):
                    if word:
                        state_ = state_ + make_label(word) + " "
                if d_index != len(all_data) - 1:
                    state_ = state_.strip() + "\n"
                f.write(state_)


# 定义HMM类，获得初始矩阵，转移矩阵，发射矩阵
class HMM:
    def __init__(self, file_text="all_train_text.txt", file_state="all_train_state.txt"):
        self.all_state = open(file_state, 'r', encoding="utf-8").read().split("\n")[:200]  # 读取标识文件
        self.all_text = open(file_text, 'r', encoding="utf-8").read().split("\n")[:200]  # 读取语料库
        self.states_to_index = {"B": 0, "M": 1, "S": 2, "E": 3}
        self.index_to_state = ["B", "M", "S", "E"]
        self.len_states = len(self.states_to_index)

        self.init_matrix = np.zeros((self.len_states))  # 初始化 初始矩阵 长度为4
        self.transfer_matrix = np.zeros((self.len_states, self.len_states))  # 初始化 转移矩阵 [4,4]
        # 发射矩阵, 使用的 2级 字典嵌套
        # # 注意这里初始化了一个  total 键 , 存储当前状态出现的总次数, 为了后面的归一化使用
        self.emit_matrix = {"B": {"total": 0}, "M": {"total": 0}, "S": {"total": 0}, "E": {"total": 0}}

    # 三个矩阵的计算函数，都是对一行数据进行处理，也就是对一篇文章进行处理，不是一次性处理所有的文字
    # 计算初始矩阵 ：统计每篇文章的第一个字的标识
    def cal_init_matrix(self, state):
        self.init_matrix[self.states_to_index[state[0]]] += 1

    # 计算转移矩阵 ： 当前状态到下一状态的次数
    def cal_transfer_matrix(self, state):
        sta_join = "".join(state)  # 变为字符串，去掉空格
        sta1 = sta_join[:-1]
        sta2 = sta_join[1:]
        for s1, s2 in zip(sta1, sta2):
            self.transfer_matrix[self.states_to_index[s1], self.states_to_index[s2]] += 1

    # 计算发射矩阵 ： 统计某种状态下，所有字出现的次数
    def cal_emit_matrix(self, words, states):
        for word, state in zip("".join(words), "".join(states)):  # 变为字符串，去掉空格
            self.emit_matrix[state][word] = self.emit_matrix[state].get(word, 0) + 1  # 统计该状态下字出现的次数
            self.emit_matrix[state]['total'] += 1  # 计算state状态下所有字出现的总次数

    # 将矩阵的值转化为概率
    def normalize(self):
        self.init_matrix = self.init_matrix / np.sum(self.init_matrix)
        self.transfer_matrix = self.transfer_matrix / np.sum(self.transfer_matrix, axis=1, keepdims=True)
        self.emit_matrix = {
            state: {word: t / word_times["total"] * 1000 for word, t in word_times.items() if word != "total"} for
            state, word_times in self.emit_matrix.items()}

    # 调用三个矩阵的计算函数，生成初始矩阵，转移矩阵，发射矩阵
    def train(self):
        # 如果三个矩阵已存在，就不训练了,直接获得三个矩阵
        if os.path.exists("three_matrix.pkl"):
            self.init_matrix, self.transfer_matrix, self.emit_matrix = pickle.load(open("three_matrix.pkl", 'rb'))
            return
        # 三个矩阵不存在
        for words, states in tqdm(zip(self.all_text, self.all_state)):  # 依次 遍历每一行的语料库和标识
            if words and states:  # 一行数据
                words = words.split(" ")  # 以空格切分每行的文字
                states = states.split(" ")  # 以空格切分每行的标识
                # 计算三大矩阵
                self.cal_init_matrix(states[0])  # 统计每篇文章的第一个字的状态
                self.cal_transfer_matrix(states)  # 统计 当前状态到下一状态的次数
                self.cal_emit_matrix(words, states)  # 统计某种状态下，所有字出现的次数
        self.normalize()  # 矩阵求完之后，将次数转化为概率
        # 保存三个矩阵
        pickle.dump([self.init_matrix, self.transfer_matrix, self.emit_matrix], open("three_matrix.pkl", 'wb'))


# 维比特算法
def viterbi_t(text, hmm):
    states = hmm.index_to_state
    # 获取训练好的三个矩阵
    emit_p = hmm.emit_matrix
    trans_p = hmm.transfer_matrix
    start_p = hmm.init_matrix
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[hmm.states_to_index[y]] * emit_p[y].get(text[0], 0)
        path[y] = [y]
    for t in range(1, len(text)):
        V.append({})
        newpath = {}

        # 检验训练的发射概率矩阵中是否有该字
        neverSeen = text[t] not in emit_p['S'].keys() and \
                    text[t] not in emit_p['M'].keys() and \
                    text[t] not in emit_p['E'].keys() and \
                    text[t] not in emit_p['B'].keys()
        for y in states:
            emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0  # 设置未知字单独成词
            temp = []
            for y0 in states:
                if V[t - 1][y0] > 0:
                    temp.append((V[t - 1][y0] * trans_p[hmm.states_to_index[y0], hmm.states_to_index[y]] * emitP, y0))
            (prob, state) = max(temp)
            # (prob, state) = max([(V[t - 1][y0] * trans_p[hmm.states_to_index[y0],hmm.states_to_index[y]] * emitP, y0)  for y0 in states if V[t - 1][y0] > 0])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    (prob, state) = max([(V[len(text) - 1][y], y) for y in states])  # 求最大概念的路径

    result = ""  # 拼接结果
    for t, s in zip(text, path[state]):
        result += t
        if s == "S" or s == "E":  # 如果是 S 或者 E 就在后面添加空格
            result += " "
    return result


if __name__ == "__main__":
    text_to_state()  # 生成隐藏状态文件
    hmm = HMM()
    hmm.train()  # 获得了三个矩阵

    # 现在开始预测
    text = "这里真的好安静啊。"
    result = viterbi_t(text, hmm)
    print(result)

