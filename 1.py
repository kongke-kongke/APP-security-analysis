import csv
import re
from androguard.core.bytecodes import apk
from androguard.core.bytecodes import dvm
from androguard.core.analysis import analysis
from androguard.decompiler import decompiler
import androguard

import networkx as nx
import  igraph as ig

import subprocess


command = "androguard cg " + " D://2.apk"+ ' -o' + ' D://cfg/2.gml'
subprocess.call(command, shell=True)

b=[]

# for i in range(11,428):
#  #command = "androguard cg " + "D://1//1 (" + '%d' % i + ').apk'+ '-o'+'D://cfg/bad/'+'%d' % i + ').gml'
#  command = "androguard cg " + " D://apk//bad//"+'%d' % i + '.apk'+ ' -o' + ' D://cfg/bad/' + '%d' % i + '.gml'
#  subprocess.call(command, shell=True)



# h=ig.Graph.Read_GML('callgraph.gml')
# g=nx.read_gml('callgraph.gml')
# a=g.nodes
# b=list(g.edges)



# with open("1.csv", "w", newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     #     # 先写入columns_name
#     #     # writer.writerow(["index", "a_name", "b_name"])
#     #     # 写入多行用writerows
#     for i in a:
#         writer.writerow([i])
# with open("2.csv", "w", newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     #     # 先写入columns_name
#     #     # writer.writerow(["index", "a_name", "b_name"])
#     #     # 写入多行用writerows
#     for i in h.get_edgelist():
#         writer.writerow(i)