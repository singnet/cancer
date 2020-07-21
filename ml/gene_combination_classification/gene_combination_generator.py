import pandas as pd
from itertools import *

# genes70 = pd.read_csv('70genes.txt')
# genes = genes70.values.tolist()
# flat_list = [item for sublist in genes for item in sublist]
# exprds = pd.read_csv('/home/oleg/Work/bc_data_procesor/git/bc_data_procesor/bmc15ANDex15bmc.csv', sep='\t')
# columnsData = exprds.reindex(flat_list, axis='columns')
# print(columnsData.shape)
# print(len(list(columnsData)))
# # columnsData = exprds.loc[ : , flat_list]
# # columnsData.to_csv('ex15bmcMerged70genes.csv', index=False, header=False)
# columnsData.to_csv('ex15bmcMerged70genes.csv')

# genes70expr = pd.read_csv('ex15bmcMerged70genes.csv', header=None)
# print(genes70expr.shape)
# expr = pd.read_csv("/home/oleg/Work/bc_data_procesor/git/bc_data_procesor/bmc15ANDex15bmc.csv", sep='\t')
# print(expr.shape)
# expr = expr.iloc[:, 0]
# print(expr.shape)
# genes70expr = genes70expr.dropna(axis=1, how='all')
# print(genes70expr.shape)
#
# comb = [list(i) for i in combinations(range(5), 2)]
# ds = genes70expr.iloc[:, comb[0]]
# print(ds.shape)
# ds = ds.to_numpy()
# datasets = [genes70expr.iloc[:, i].to_numpy() for i in comb]


class CombinationsGenerator:

    def __init__(self, path_to_a_dataset):
        self.dataset = pd.read_csv(path_to_a_dataset).dropna(axis=1, how='all')

    def generate(self, num_of_items):
        num_of_genes = self.dataset.shape[1]
        comb = [list(i) for i in combinations(range(num_of_genes), num_of_items)]
        col_names_lst = list(self.dataset)
        head = col_names_lst.pop(0)
        gene_names_comb = [list(i) for i in combinations(col_names_lst, num_of_items)]
        subsets = [self.dataset.iloc[:, i].to_numpy() for i in comb]
        return subsets, gene_names_comb

# comb_gen = CombinationsGenerator("ex15bmcMerged70genes.csv")
# datasets, gene_names_list = comb_gen.generate(2)
# print(datasets[0].shape)
# print(len(gene_names_list), gene_names_list)


