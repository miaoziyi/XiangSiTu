
#!/usr/bin/env python
# -*- coding:utf-8 -*-

def file_same():
    str1 = []
    file1 = open("query-imgPath_new.txt", "r", encoding="utf-8")
    for line in file1.readlines():  # 读取第一个文件
        str1.append(line.split('/')[-1].replace("\n", ""))

    str2 = []
    file2 = open(r"D:\xiangsitu\related/query_img_2.txt", "r", encoding="utf-8")
    for line in file2.readlines():  # 读取第二个文件
        str2.append(line.split('/')[-1].replace("\n", ""))

    str_dump = []
    a = 0
    for line in str1:
        if line in str2:
            str_dump.append(line)  # 将两个文件重复的内容取出来
            print(line)  # 将重复的内容输出
            a = a+1
            print("*"*80)

    str_all = set(str1 + str2)  # 将两个文件放到集合里，过滤掉重复内容
    # print(a)

    for i in str_dump:
        if i in str_all:
            str_all.remove(i)  # 去掉两个文件中重复的内容

    for str in str_all:  # 去重后的结果写入文件
        # print(str)
        with open("imgPath_new_error.txt", "a+", encoding="utf-8") as f:
            f.write(str + "\n")


if __name__ == "__main__":
    file_same()
