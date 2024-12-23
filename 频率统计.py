import csv
import heapq
import glob
import networkx as nx
import  igraph as ig
import numpy as np
import csv
path=glob.glob("D://cfg/begin//*.gml")
for k in path:
    h=ig.Graph.Read_GML(k)
    g=nx.read_gml(k)
    e=list(g.nodes)
    a=h.get_edgelist()
    print(len(a),len(e))
    b=[]
    for i in range(len(a)):
        if a[i][0] not in b:
            b.append(a[i][0])
    c=[0]*len(b)
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i][0] == b[j]:
                c[j]=c[j]+1
    #输出C的全部值，用来设计删除攻击的复原方法
    sum=0
    for i in c:
        sum=sum+c[i]
        # print(c[i])
    print(sum,sum/len(c))


    # c_max = heapq.nlargest(10,c)
    # temp=[]
    # for i in c_max:
    #     if i not in temp:
    #         temp.append(i)
    # index_max=[]
    # for j in temp:
    #     for i in range(len(c)):
    #       if j==c[i]:
    #           index_max.append(i)

    # for i in index_max:
    #     z=b[i]
    #     print(e[z]+',',c[i])

  #被调用
    # d = [0] * len(b)
    # for i in range(len(a)):
    #     for j in range(len(b)):
    #         if a[i][1] == b[j]:
    #             d[j] = d[j] + 1
    # d_max = heapq.nlargest(10, d)
    # temp = []
    # for i in d_max:
    #     if i not in temp:
    #         temp.append(i)
    # index_maxd = []
    # for j in temp:
    #     for i in range(len(d)):
    #         if j == d[i]:
    #             index_maxd.append(i)
    #
    # for i in index_maxd:
    #     z=b[i]
    #     print(e[z]+',',d[i])







# with open("pinlv.csv", "w", newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     #     # 先写入columns_name
#     #     # writer.writerow(["index", "a_name", "b_name"])
#     #     # 写入多行用writerows
#     for i in b:
#         writer.writerow([i])
# with open("pinlv2.csv", "w", newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     #     # 先写入columns_name
#     #     # writer.writerow(["index", "a_name", "b_name"])
#     #     # 写入多行用writerows
#     for i in c:
#         writer.writerow([i])

