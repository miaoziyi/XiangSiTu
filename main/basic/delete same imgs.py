# 删除两个文件夹下同名文件
import os


def delect(dir1, dir2):
    list2 = os.listdir(dir2)
    list3 = []
    for i in list2:
        list3.append(i)

    list1 = os.listdir(dir1)
    for i in list1:
        if i in list3:
            os.remove(dir1 + '\\' + i)
        else:
            continue


if __name__ == '__main__':
    dir_big = r"D:\xiangsitu\img_test_cid2\11\1102"
    dir_small = r"D:\xiangsitu\img_test\11\1103\110301"
delect(dir_big, dir_small)
