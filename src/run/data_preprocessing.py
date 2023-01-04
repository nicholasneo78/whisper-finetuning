# imports
import os
import pandas as pd
import json
from tqdm import tqdm
from typing import Tuple
from modules import preprocess_text

class GeneratePickleFromManifest:
    '''
        generate the pkl from manifest with all the data required to build the DatasetDict for the finetuning step 
    '''
    def __init__(self, manifest_path: str, pkl_filename: str, label: str, language: str='en', additional_preprocessing: str='general') -> None:
        '''
            manifest_path: the path to retrieve the manifest file with the information of the audio path and annotations
            pkl_filename: the file path where the pickle data file will reside after preprocessing
            additional_preprocessing: depending on the annotations given, are there any other additional preprocessing needed to standardize the annotations
            label: label of the data
            language: language of the data
        '''
        self.manifest_path = manifest_path
        self.pkl_filename = pkl_filename
        self.additional_preprocessing = additional_preprocessing
        self.label = label
        self.language = language

    # def create_new_dir(self, directory: str) -> None:
    #     '''
    #         create new directory and ignore already created ones

    #         directory: the directory path that is being created
    #     '''

    #     try:
    #         os.mkdir(directory)
    #     except OSError as error:
    #         pass # directory already exists!

    # def preprocess_text(self, text: str) -> str:
    #     '''
    #         all the text preprocessing being done for the annotations

    #         text: text annotations required to be preprocessed            
    #     '''

    #     # additional preprocessing to replace the filler words with one symbol
    #     if self.additional_preprocessing == 'general':
    #         clean_text = text.replace('#', ' ').replace('<unk>', '#')
        
    #     # add more here for other filler word or additional preprocessing needed for other data
    #     # elif ...

    #     else:
    #         clean_text = text.replace('#', ' ')
        
    #     # keep only certain characters
    #     clean_text = re.sub(r'[^A-Za-z0-9#\' ]+', ' ', clean_text)
        
    #     # replace hyphen with space because hyphen cannot be heard
    #     clean_text = clean_text.replace('-', ' ')

    #     # convert all the digits to its text equivalent
    #     clean_text = get_text_from_number(text=clean_text, 
    #                                       label=self.label, 
    #                                       language=self.language)
        
    #     # convert multiple spaces into only one space
    #     clean_text = ' '.join(clean_text.split())

    #     # returns the preprocessed text
    #     return clean_text.upper()

    def build_pickle_from_manifest(self) -> Tuple[pd.DataFrame, str]:
        '''
            generate the pickle file from manifest data file to prepare the final dataset for finetuning step
        '''

        # dict_list: to create a list to store the dictionaries from the manifest file
        # data_list: to store the data into this list to be exported into a pkl file
        dict_list, data_list = [], []

        # load manifest file into a list of dictionaries
        with open(self.manifest_path, 'rb') as f:
            for line in f:
                dict_list.append(json.loads(line))

        # iterate through the data_list and create the final pkl dataset file
        for entries in tqdm(dict_list):

            # get the array of values from the audio files and using 16000 sampling rate (16000 due to w2v2 requirement)
            # split the rightmost / and only take the parent directory of the manifest file
            # audio_array, _ = librosa.load(f"{self.manifest_path.rsplit('/', 1)[0]}/{entries['audio_filepath']}", sr=16000)

            # text preprocessing
            clean_text = preprocess_text(
                text=entries['text'],
                label=self.label,
                language=self.language
            )

            # creating the final data dictionary that is to be saved to a pkl file
            data = {'file': f"{self.manifest_path.rsplit('/', 1)[0]}/{entries['audio_filepath']}",
                    'audio': {
                        # 'array': audio_array,
                        'path': f"{self.manifest_path.rsplit('/', 1)[0]}/{entries['audio_filepath']}",
                        'sampling_rate': 16000
                    },
                    'text': clean_text,
                    'label': self.label
            }

            data_list.append(data)

        # form the dataframe
        df_final = pd.DataFrame(data_list)

        # export the dataframe to pickle
        df_final.to_pickle(self.pkl_filename)

        # returns the final preprocessed dataframe and the filepath of the pickle file
        return df_final, self.pkl_filename
    

    def __call__(self):
        return self.build_pickle_from_manifest()


if __name__ == "__main__":

    librispeech_train_pkl = GeneratePickleFromManifest(manifest_path='/whisper_finetuning/datasets/librispeech/librispeech-100/train_manifest_small.json', 
                                                       pkl_filename='/whisper_finetuning/datasets/librispeech/train_small.pkl',
                                                       label='librispeech',
                                                       language='en')

    librispeech_dev_pkl = GeneratePickleFromManifest(manifest_path='/whisper_finetuning/datasets/librispeech/librispeech-dev-clean/dev_manifest_small.json', 
                                                     pkl_filename='/whisper_finetuning/datasets/librispeech/dev_small.pkl',
                                                     label='librispeech',
                                                     language='en')

    librispeech_test_pkl = GeneratePickleFromManifest(manifest_path='/whisper_finetuning/datasets/librispeech/librispeech-test-clean/test_manifest.json', 
                                                       pkl_filename='/whisper_finetuning/datasets/librispeech/test.pkl',
                                                       label='librispeech',
                                                       language='en')

    _, _ = librispeech_train_pkl()
    _, _ = librispeech_dev_pkl()
    _, _ = librispeech_test_pkl()