import numpy as np

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



def main():
    filename = "matrix_total.npy"
    increment = 500

    origin = np.load(filename, mmap_mode="c")
    n_samples = printSetInfo(origin, filename)

if __name__ == '__main__':
    main()