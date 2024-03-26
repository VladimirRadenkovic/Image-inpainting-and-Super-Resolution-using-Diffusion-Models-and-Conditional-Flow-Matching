# https://github.com/Ovikx/PyTorch-Checkpoint-Manager/blob/master/LICENSE
import os
import torch

class CheckpointManager:
    def __init__(self, assets, directory, file_name, maximum=3, file_format='pt'):
        '''
        Constructs the CheckpointManager

        Args:
            assets : Dictionary, the states you want to save or load
            directory : String, the path the states will be saved to or loaded from
            file_name : String, the name of the saved file
            maximum : Integer, the maximum number of saves to keep
            file_format : String, the file format of the saved file
        '''
        self.assets = assets
        self.directory = directory
        if self.directory[-1] != '/':
            self.directory += '/'
        self.file_name = file_name
        self.maximum = maximum
        self.file_format = file_format
    
    def index_from_file(self, file_name):
        return int(file_name[len(self.file_name)+1:-(len(self.file_format)+1)])
    
    def save(self, epoch=None):
        '''
        Saves the asset dictionary

        Args:
            epoch : Integer, a custom index to concatenate to the end of the file name (intended for epoch numbers)
        '''
        dir_contents = os.listdir(self.directory)

        if len(dir_contents) > 0 and epoch == None:
            index = max([self.index_from_file(file_name) for file_name in dir_contents]) + 1
        else:
            index = epoch if epoch != None else 1
        
        save_dir = f'{self.directory}{self.file_name}_{index}.{self.file_format}'
        torch.save(self.assets, save_dir)
        
        self.purge()
        print(f'Saved states to {save_dir}')
        
    def purge(self):
        dir_contents = os.listdir(self.directory)
    
        if len(dir_contents) > self.maximum:
            indices = sorted([self.index_from_file(v) for v in dir_contents])
            for index in indices[:len(indices)-self.maximum]:
                for directory in dir_contents:
                    if f'{index}.{self.file_format}' in directory:
                        os.remove(f'{self.directory}{directory}')
    
    def load(self):
        '''
        Returns the state dictionary of the highest indexed file in the save directory. Returns constructor asset input if no such file exists.
        '''
        dir_contents = os.listdir(self.directory)

        for directory in dir_contents:
            if self.file_name in directory:
                max_index = max([self.index_from_file(v) for v in dir_contents])
                load_dir = f'{self.directory}{self.file_name}_{max_index}.{self.file_format}'
                print(f'Loading states from {load_dir}')
                return torch.load(load_dir)
        
        print('Initializing fresh states')
        return self.assets