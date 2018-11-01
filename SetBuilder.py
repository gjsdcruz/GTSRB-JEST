import numpy as np
import os
import pandas as pd
#import sklearn as sk
from sklearn.feature_selection import SelectKBest, chi2
from scipy import stats

def get_Samples_of(dset):
    return dset.shape[0]

def printSetInfo(dset, name):
    '''
    Prints dataset shape in an easier to read format
    Parameters
    ----------
    dset : array-like object
        This is the dataset to be described.
    name : str
        Dataset's original filename
    Returns
    -------
    out : int
        Returns the number of samples in dset. This is
        used because more often than not this value needs
        to be used after printing the dataset info.
    '''
    print("\nCurrent file name: ", name)
    print("Number of samples: " + str(dset.shape[0]))
    print("Features per sample: " + str(dset.shape[1]) + "\n")
    return get_Samples_of(dset)

def read_npy_chunk(filename, start_row, num_rows):
    """
    Reads a partial array (contiguous chunk along the first
    axis) from an NPY file.
    Parameters
    ----------
    filename : str
        Name/path of the file from which to read.
    start_row : int
        The first row of the chunk you wish to read. Must be
        less than the number of rows (elements along the first
        axis) in the file.
    num_rows : int
        The number of rows you wish to read. The total of
        start_row + num_rows must be less than the number of
        rows (elements along the first axis) in the file.
    Returns
    -------
    out : ndarray
        Array with out.shape[0] == num_rows, equivalent to
        `arr[start_row:start_row + num_rows]` if `arr` were
        the entire array (note that the entire array is never
        loaded into memory by this function).
    """
    assert start_row >= 0 and num_rows > 0
    with open(filename, 'rb') as fhandle:
        major, minor = np.lib.format.read_magic(fhandle)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fhandle)
        assert not fortran, "Fortran order arrays not supported"
        # Make sure the offsets aren't invalid.
        assert start_row < shape[0], (
            'start_row is beyond end of file'
        )
        assert start_row + num_rows <= shape[0], (
            'start_row + num_rows > shape[0]'
        )
        # Get the number of elements in one 'row' by taking
        # a product over all other dimensions.
        row_size = np.prod(shape[1:])
        start_byte = start_row * row_size * dtype.itemsize
        fhandle.seek(start_byte, 1)
        n_items = row_size * num_rows
        flat = np.fromfile(fhandle, count=n_items, dtype=dtype)
        return flat.reshape((-1,) + shape[1:])

"""
def merge_csvs_into(csv_name, prefix):
    '''
    Merges various csvs (must be saved in the script's directory) into one. 
    The csvs to be merged must have a format to their filename
    e.g.: (FILE1.csv, FILE2.csv),(FOO1.csv, FOO2.csv, FOO3.csv), etc...
    Parameters
    ----------
    csv_name : str
        Specifies the name of the folder to be created
    prefix : str
        Specifies what numbered files to merge. Example: if the
        files to merge are formatted as ("file1.csv","file2.csv",
        file3.csv,...,filex.csv), prefix must be "file"
    Returns
    -------
    This function does not return anything. 
    '''
    csv_name = csv_name + ".csv"
    fout=open(csv_name,"a")
    # first file:
    for line in open(prefix + ".csv"):
        fout.write(line)
    # now the rest:    
    for num in range(2,201):
        fname = prefix+str(num)+".csv"
        f = open(fname)
        f.next() # skip the header
        for line in f:
            fout.write(line)
        os.remove(fname)
    fout.close()


def buffered_Dataset_Cleaner(filename,n_samples,increment=100):
    '''
    Reads a dataset file through several partial arrays into a buffer, 
    which it then cleans and saves into a .csv file. After saving, reads
    the next partial array until there are none left, at which point
    merges and deletes all csvs.
    Parameters
    ----------
    n_samples : int
        Number of samples in the original dataset
    incr : int
        Number of samples to read into the buffer each time. 
            -smaller value = larger overhead and more operations.
            -larger value = larger memory consumption.
    filename : str
        Name of the original dataset's file.
    Returns
    -------
    This function doesn't return anything
    '''
    start_row=0
    current_csv = 1
    while start_row < n_samples:
        data = read_npy_chunk(filename, start_row, increment)
        data = dropna(data)
        np.savetxt("MAT"+str(current_csv)+".csv",data,delimiter=",")
        current_csv = current_csv + 1
        start_row = start_row + increment
        if start_row == n_samples/increment*increment: start_row = start_row + n_samples%increment

    merge_csvs_into("Matrix_modified","MAT")
"""
def dropna(data):
    return data[~np.isnan(data)]

def main():
    filename = "matrix_total.npy"
    increment = 500

    origin = np.load(filename, mmap_mode="c")
    n_samples = printSetInfo(origin, filename)

    #print("Applying zscore to dataset for normalization\n")
    #origin = stats.zscore(origin)
    #print("Dataset normalized")


    classSizes = [211, 2221, 2251, 1441, 1981, 1861, 421, 1441, 1411, 1471, 2011, 1321, 2101, 2161, 781, 631,
                421, 1111, 1201, 211, 361, 331, 391, 511, 271, 1501, 601, 241, 541, 271, 451, 781, 241, 690, 421, 1201,
                391, 211, 2071, 301, 361, 241, 241]


    new_data = []
    lastIndex = 0
    for i in range(42):
        featureClass = origin[lastIndex:lastIndex+210,:]
        for k in range(210):
            new_data.append(featureClass[k,:])
        lastIndex = classSizes[i]-1
        print("Sampling class "+ str(i+1)+" of 43 total\n")
    featureClass = origin[lastIndex:lastIndex+210,:]
    for i in range(210):
        new_data.append(featureClass[i,:])
    print("Sampling class 43 of 43 total\n")

    dset = np.array(new_data)
    N, M = dset.shape
    M = M+1
    data = np.zeros((N,M))
    data[:,:-1] = dset
    printSetInfo(data, filename)
    return
    for i in range(1,42):
        for k in range(210):
            data[i*210+k,M]=i
            print("Labelling class "+ str(i+1)+" of 43 total\n")

    print("Selecting features through chi2 statistical test\n")
    data = SelectKBest(chi2,k=15).fit_transform(data[:,:-2],data[:,-1])
    print("Features selected!\n")

    printSetInfo(data, filename)

    np.save(filename, data)




    
if __name__ == '__main__':
    main()