!pip install mlxtend
!pip install apyori graphviz

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
import graphviz

#Create a sample dataset:
data = [['Timestream', 'Temperature', 'Energy_consumption'],
        ['Timestream', 'Temperature'],
        ['Timestream', 'Humidity'],
        ['Temperature', 'Humidity'],
        ['Timestream', 'Temperature', 'Humidity', 'Energy_consumption'],
        ['Holiday', 'Temperature', 'Humidity']]

df = pd.DataFrame(data, columns=['Item1', 'Item2', 'Item3', 'Item4'])
df

#Convert the dataset to a transaction format:
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
df_encoded


#Apply the Apriori algorithm:
frequent_itemsets_apriori = apriori(df_encoded, min_support=0.33, use_colnames=True)
frequent_itemsets_apriori

#Apply the FP-growth algorithm:
frequent_itemsets_fpgrowth = fpgrowth(df_encoded, min_support=0.33, use_colnames=True)
frequent_itemsets_fpgrowth

#Construction of FP-Tree
class Node:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}

def build_tree(data, min_support):
    header_table = {}
    for index, row in data.iterrows():
        for item in row:
            header_table[item] = header_table.get(item, 0) + 1

    for k in list(header_table):
        if header_table[k] < min_support:
            del header_table[k]

    frequent_items = list(header_table.keys())
    frequent_items.sort(key=lambda x: header_table[x], reverse=True)

    root = Node("Null", 1, None)

    for index, row in data.iterrows():
        ordered_items = [item for item in frequent_items if item in row]
        if ordered_items:
            insert_tree(ordered_items, root, header_table, 1)

    # Ensure 'Null' is in header_table
    if 'Null' not in header_table:
        header_table['Null'] = (0, None)

    return root, header_table

def insert_tree(items, node, header_table, count):
    if not items:
        return

    if items[0] in node.children:
        node.children[items[0]].count += count
    else:
        node.children[items[0]] = Node(items[0], count, node)

        if header_table[items[0]][1] is None:
            header_table[items[0]] = (header_table[items[0]][0], node.children[items[0]])
        else:
            update_header(header_table[items[0]][1], node.children[items[0]])

    if len(items) > 1:
        insert_tree(items[1:], node.children[items[0]], header_table, count)

def update_header(node_to_test, target_node):
    while node_to_test.nodeLink is not None:
        node_to_test = node_to_test.nodeLink
    node_to_test.nodeLink = target_node

# FP-tree construction
root, header_table = build_tree(df, min_support=2)

# Visualize the FP-tree
def visualize_tree(node, graph, parent_name, graph_name):
    if node is not None:
        graph.node(graph_name, f"{node.item} ({node.count})", shape="box")
        if parent_name is not None:
            graph.edge(parent_name, graph_name)
        for child_key, child_node in node.children.items():
            visualize_tree(child_node, graph, graph_name, f"{graph_name}_{child_key}")

# Create a graph using Graphviz
fp_tree_graph = graphviz.Digraph('FP_Tree', node_attr={'shape': 'box'}, graph_attr={'rankdir': 'TB'})
visualize_tree(root, fp_tree_graph, None, 'Root')

# Display the FP-tree visualization
fp_tree_graph.render(filename='fp_tree_visualization', format='png', cleanup=True)
fp_tree_graph

#Answer the questions based on the FP-tree:
# a) Find maximum frequent itemset
max_frequent_itemset_fp = frequent_itemsets_fpgrowth[frequent_itemsets_fpgrowth['support'] == frequent_itemsets_fpgrowth['support'].max()]
print("a) Maximum Frequent Itemset (FP-growth):\n", max_frequent_itemset_fp)

max_frequent_itemset_apriori = frequent_itemsets_apriori[frequent_itemsets_apriori['support'] == frequent_itemsets_apriori['support'].max()]
print("a) Maximum Frequent Itemset (Apriori):\n", max_frequent_itemset_apriori)

# b) How many transactions does it contain?
num_transactions = len(df)
print("b) Number of transactions in the dataset:", num_transactions)

# c) Simulate frequent pattern enumeration based on the FP-tree constructed.
def mine_patterns(node, prefix, header_table, min_support, patterns):
    if header_table[node.item][0] >= min_support:
        patterns.append(prefix + [node.item])

    for child_key, child_node in node.children.items():
        mine_patterns(child_node, prefix + [node.item], header_table, min_support, patterns)


patterns_apriori = list(frequent_itemsets_apriori['itemsets'])
print("c) Frequent Patterns Enumerated (Apriori):\n", patterns_apriori)
patterns_fp = []
mine_patterns(root, [], header_table, min_support=2, patterns=patterns_fp)
print("c) Frequent Patterns Enumerated (FP-growth):\n", patterns_fp)


