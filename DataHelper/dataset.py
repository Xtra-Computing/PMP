import abc


class dataset:
    dataset_name = None
    dataset_descrition = None
    
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    data = None
    
    
    def __init__(self, dName=None, dDescription=None):
        self.dataset_name = dName
        self.dataset_descrition = dDescription
    
    def print_dataset_information(self):
        print('Dataset Name: ' + self.dataset_name)
        print('Dataset Description: ' + self.dataset_descrition)

    @abc.abstractmethod
    def load(self):
        return
    