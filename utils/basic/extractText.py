className = '17'
filename = "../../img1_0/train/result_1.txt"
outfilename = "D:\\mzy\\img_test_17\\17.txt"
with open(filename, mode="r", encoding="utf-8") as f:
    with open(outfilename, "w", encoding='utf-8') as f1:
        res = f.readlines()
        print('txt总行数')
        print(len(res))
        for s in res:
            x = s.split(',')
            # 注意!!!!!原本txt包含内容可能不同   [4]代表二级分类[5]三级分类
            if x[2] == className:
                f1.write(s)
        f1.close()
        f.close()
print('写入完成...')
