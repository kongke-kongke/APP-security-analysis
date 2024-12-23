import networkx as nx
import  igraph as ig
import csv
h=ig.Graph.Read_GML('callgraph.gml')
g=nx.read_gml('callgraph.gml')
a=g.nodes
a=list(a)
c=h.get_edgelist()
print(a[0])
print(c[0],c[0][0])

# b=list(g.edges)


#
# with open("1.csv", "w", newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     #     # 先写入columns_name
#     #     # writer.writerow(["index", "a_name", "b_name"])
#     #     # 写入多行用writerows
#     for i in a:
#         writer.writerow([i])
#
# with open("2.csv", "w", newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     #     # 先写入columns_name
#     #     # writer.writerow(["index", "a_name", "b_name"])
#     #     # 写入多行用writerows
#     for i in h.get_edgelist():
#         writer.writerow(i)