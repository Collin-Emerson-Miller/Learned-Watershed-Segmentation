import datasetutils
import os
import h5py


cremi_train_urls = ["https://cremi.org/static/data/sample_A_20160501.hdf",
              "https://cremi.org/static/data/sample_B_20160501.hdf",
              "https://cremi.org/static/data/sample_C_20160501.hdf"]

cremi_test_urls = ["https://cremi.org/static/data/sample_A%2B_20160601.hdf",
              "https://cremi.org/static/data/sample_B%2B_20160601.hdf",
              "https://cremi.org/static/data/sample_C%2B_20160601.hdf"]

cremi_path = "cremi"

def get_data(dataset=0):
    """Returns the raw data and neuron ids.
    
    dataset 0 = A
    dataset 1 = B
    dataset 2 = c
    
    """
    while os.getcwd().split(os.sep)[-1] != "Learned-Watershed-Segmentation":
        os.chdir("..")
        cwd = os.getcwd()
    os.chdir(os.path.join("segmentation", "datasets"))

    if not os.path.exists(cremi_path):
        os.mkdir(cremi_path)
    
    url = cremi_train_urls[dataset]
    
    file_name = os.path.join(cremi_path, url.split('/')[-1])

    if not os.path.exists(file_name) or (datasetutils.get_url_file_size(url) != os.path.getsize(file_name)):
        print("Downloading File")
        datasetutils.download_file(url, file_name)

    # Get neuron ids from Cremi data.
    with h5py.File(file_name, "r") as hdf:
        print ("Initializing...")
        labels = hdf['volumes']['labels']['neuron_ids'][:]
        raw = hdf['volumes']['raw'].value
        print ("Done!")

        return raw, labels