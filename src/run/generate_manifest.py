# imports
import os
from os.path import join
import numpy as np
import pandas as pd
import json
import librosa
from pathlib import Path

# build a class to produce the librispeech data manifest
class GenerateManifest:
    
    def __init__(self, root_folder, manifest_filename, got_annotation, audio_ext):
        self.root_folder = root_folder
        self.manifest_filename = manifest_filename
        self.got_annotation = got_annotation
        self.audio_ext = audio_ext
    
    # check if the json file name already existed (if existed, need to throw error or else the new json manifest will be appended to the old one, hence causing a file corruption)
    def json_existence(self):
        assert not os.path.isfile(f'{self.manifest_filename}'), "json filename exists! Please remove old json file!"
    
    # helper function to build the lookup table for the id and annotations from all the text files and return the table
    def build_lookup_table(self):
        #initiate list to store the id and annotations lookup
        split_list_frame = []

        # get all the annotations into a dataframe
        for root, subdirs, files in os.walk(self.root_folder):
            for file in files:
                if file.endswith(".txt"):
                    # add on to the code here
                    df = pd.read_csv(os.path.join(root, file), header=None)
                    df.columns = ['name']

                    for i,j in enumerate(df.name):
                        split_list = j.split(" ",1)
                        split_list_frame.append(split_list)

        df_new = pd.DataFrame(split_list_frame, columns=['id', 'annotations']) # id and annotations are just dummy headers here
        return df_new
    
    # helper function to create the json manifest file
    def create_json_manifest(self):
        
        # check if the json filename have existed in the directory
        self.json_existence()
        
        if self.got_annotation:
            # get the lookup table
            df_new = self.build_lookup_table()

        error_count = 0

        # retrieve the dataframe lookup table
        for root, subdirs, files in os.walk(self.root_folder):
            # since self.root_folder is a subset of the root, can just replace self.root with empty string
            modified_root_ = str(Path(root)).replace(str(Path(self.root_folder)), '')
            # replace the slash with empty string after Path standardization
            modified_root = modified_root_.replace('/', '', 1)

            for file in files:
                if file.endswith(self.audio_ext):
                    try:
                        # retrieve the base path for the particular audio file
                        base_path = os.path.basename(os.path.join(root, file)).split('.')[0]

                        # create the dictionary that is to be appended to the json file
                        if self.got_annotation:
                            data = {
                                    'audio_filepath' : os.path.join(modified_root, file),
                                    # 'audio_filepath' : os.path.join(root, file),
                                    'duration' : librosa.get_duration(filename=os.path.join(root, file)),
                                    'text' : df_new.loc[df_new['id'] == base_path, 'annotations'].to_numpy()[0]
                                }
                        else:
                            data = {
                                    'audio_filepath' : os.path.join(modified_root, file),
                                    # 'audio_filepath' : os.path.join(root, file),
                                    'duration' : librosa.get_duration(filename=os.path.join(root, file)),
                                    'text': ""
                                }

                        # write to json file
                        with open(f'{self.manifest_filename}', 'a+', encoding='utf-8') as f:
                            f.write(json.dumps(data) + '\n')
            
                    # for corrupted file of the target extension                
                    except:
                        error_count+=1
                        print(f"Error loading {file}")
                        continue
        
        print(f'Total number of errors: {error_count}')
        return f'{self.manifest_filename}'

    def __call__(self):
        return self.create_json_manifest()

if __name__ == '__main__':

    # get_manifest_train = GenerateManifest(root_folder="/manifest_preprocessing/datasets/librispeech/librispeech-100/", 
    #                                       manifest_filename="/manifest_preprocessing/datasets/librispeech/librispeech-100/train_manifest.json", 
    #                                       got_annotation=True,
    #                                       audio_ext='.wav')
    # _ = get_manifest_train()

    get_manifest_dev = GenerateManifest(root_folder="/manifest_preprocessing/datasets/librispeech/librispeech-dev-clean/", 
                                        manifest_filename="/manifest_preprocessing/datasets/librispeech/librispeech-dev-clean/dev_manifest.json", 
                                        got_annotation=True,
                                        audio_ext='.wav')
    _ = get_manifest_dev()

    get_manifest_test = GenerateManifest(root_folder="/manifest_preprocessing/datasets/librispeech/librispeech-test-clean/", 
                                         manifest_filename="/manifest_preprocessing/datasets/librispeech/librispeech-test-clean/test_manifest.json", 
                                         got_annotation=True,
                                         audio_ext='.wav')
    _ = get_manifest_test()