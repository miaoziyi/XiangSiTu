import os

ff = open('query-imgPath3.txt', 'w')


# 查找指定文件夹下所有相同名称的文件
def search_file(dirPath, fileName, ff):
    dirs = os.listdir(dirPath)  # 查找该层文件夹下所有的文件及文件夹，返回列表
    for currentFile in dirs:  # 遍历列表
        absPath = dirPath + '/' + currentFile
        if os.path.isdir(absPath):  # 如果是目录则递归，继续查找该目录下的文件
            search_file(absPath, fileName, ff)
        elif currentFile == fileName:
            ff.write(absPath+'\n')
            # return absPath  # 文件存在，则打印该文件的绝对路径


if __name__ == "__main__":
    dirPath = 'D:\\xiangsitu\\img_test\\new'
    with open('../txt/query-imgPath2.txt', 'r', encoding='utf8') as file:
        imgs = file.readlines()
        for i, img in enumerate(imgs):
            # 图片存在
            img = img.split('/')[-1].replace('\n', '')
            search_file(dirPath, img, ff)
            # ff.write(path+'\n')
