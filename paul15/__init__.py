import scipy.sparse
import scipy.io
import numpy as np
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize

# Original Labels are within 0 to 9. But proper label mapping is required as there are 8 classes.

class DataSampler(object):
    def __init__(self):
        self.total_size = 2730
        self.X_train, self.y_train, self.mapping = self._load_data()
        
    def _load_data(self):    
        #we have downloaded this data, if not, the following will download data automatically
        adata=sc.datasets.paul15()
        adata.obs['celltype']=adata.obs['paul15_clusters'].str.split("[0-9]{1,2}", n = 1, expand = True).values[:,1]
        adata.obs['celltype2']=adata.obs['paul15_clusters']
        sc.pp.log1p(adata)
        #sc.pp.filter_genes_dispersion(adata,n_top_genes=1000) #older scanpy
        sc.pp.highly_variable_genes(adata,n_top_genes=1000,subset=True,inplace=True)
        cell_type = np.unique(adata.obs['celltype'])
        idxes = [i for i in range(10)]
        mapping = dict(zip(cell_type, idxes))
        y = [mapping[i] for i in adata.obs['celltype']]
        y = np.array(y).astype(int)
        return adata.X, y, mapping

       
    def _load_h5(self, filename):
        data_mat = h5py.File(filename)
        x = np.array(data_mat['X'])
        y = np.array(data_mat['Y']).astype(int)
        
        # preprocessing scRNA-seq read counts matrix
        adata = sc.AnnData(x)
        adata.obs['Group'] = y
    
        adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)
    
        adata = normalize(adata,
                          size_factors=True,
                          normalize_input=True,
                          logtrans_input=True)
    
        input_size = adata.n_vars
    
        print(adata.X.shape)
        print(y.shape)
        x_sd = adata.X.std(0)
        x_sd_median = np.median(x_sd)
        print("median of gene sd: %.5f" % x_sd_median)
        return adata.X, y
       
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)

        if label:
            return self.X_train[indx, :], self.y_train[indx].flatten()
        else:
            return self.X_train[indx, :]

    def validation(self):
        return self.X_train, self.y_train

    def test(self):
        return self.X_train, self.y_train

    def load_all(self):
         #return np.concatenate((self.X_train, self.X_test)), np.concatenate((self.y_train, self.y_test))
        return self.X_train, self.y_train
