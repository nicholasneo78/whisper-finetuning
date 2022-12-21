from tqdm import tqdm
import ipywidgets
import torch
import torchaudio
import librosa
import pickle
import pandas as pd
import accelerate
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple
import datasets
from datasets import Dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.integrations import TensorBoardCallback
#import evaluate
from jiwer import compute_measures

class WER(datasets.Metric):
    '''
        WER metrics
    '''

    def __init__(self, predictions=None, references=None, concatenate_texts=False):
        self.predictions = predictions
        self.references = references
        self.concatenate_texts = concatenate_texts

    def compute(self):
        if self.concatenate_texts:
            return compute_measures(self.references, self.predictions)['wer']
        else:
            incorrect = 0
            total = 0
            for prediction, reference in zip(self.predictions, self.references):
                measures = compute_measures(reference, prediction)
                incorrect += measures['substitutions'] + measures['deletions'] + measures['insertions']
                total += measures['substitutions'] + measures['deletions'] + measures['hits']
            return incorrect/total

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    '''
        the collator class to collate the data for finetuning the whisper model
    '''

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class WhisperFinetuning:
    '''
        the main class to do the whisper finetuning on another dataset (with data swapping capabilities)
    '''

    def __init__(
        self, 
        train_pkl_dir: str, 
        dev_pkl_dir: str, 
        test_pkl_dir: str, 
        root_path_to_be_removed: str, 
        root_path_to_be_replaced: str,
        pretrained_whisper_model_dir: str,
        finetuned_language: str,
        finetuned_output_dir: str,
        num_processes: str,
        learning_rate: str,
        weight_decay: str,
        warmup_steps: str,
        num_train_epochs: str,
        save_eval_logging_steps: str
    ) -> None:
        '''
            train_pkl_dir: path to the train data pickle file
            dev_pkl_dir: path to the dev data pickle file
            test_pkl_dir: path to the test data pickle file
            root_path_to_be_removed: the original old root path to be replaced in the pickle file
            root_path_to_be_replaced: the new root path to replace the old one in the pickle file
            pretrained_whisper_model_dir: the pretrained whisper model's directory that will be finetuned on
            finetuned_language: the lanuguage that the pretrained model is going to finetune on
            finetuned_output_dir: the path to where the final saved model will be stored
            num_processes: how many gpus to use for the training, 1,2 or 4 for distributed training
            learning_rate: how fast the gradient of the model will descent
            weight_decay:  how fast the learning rate will decay every epoch
            warmup_steps: how many steps with a lower learning rate to get the training warmed up
            num_train_epochs: number of epochs in the training
            save_eval_logging_steps: number of steps interval to save, evaluate and log the steps, produces the log, checkpoint models and do an evaluation on the dev/validation set
        '''
        
        self.train_pkl_dir = train_pkl_dir
        self.dev_pkl_dir = dev_pkl_dir
        self.test_pkl_dir = test_pkl_dir
        self.root_path_to_be_removed = root_path_to_be_removed
        self.root_path_to_be_replaced = root_path_to_be_replaced
        self.pretrained_whisper_model_dir = pretrained_whisper_model_dir
        self.finetuned_language = finetuned_language
        self.finetuned_output_dir = finetuned_output_dir
        self.num_processes = num_processes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_train_epochs = num_train_epochs
        self.save_eval_logging_steps = save_eval_logging_steps

        # initialise the loading of the feature extractor, tokenizer and the processor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.pretrained_whisper_model_dir)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.pretrained_whisper_model_dir, language=self.finetuned_language, task="transcribe")
        #self.processor = WhisperProcessor.from_pretrained(self.pretrained_whisper_model_dir, language=self.finetuned_language, task="transcribe")

        # initalise the metrics
        # self.metric = evaluate.load("wer")
        
    def load_audio_tensor_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
            to iterate the the dataframe to store the audio array that is required by the whisper dataset format

            df: the dataframe passed in to retrieve the audio array based on the filepath of the audio
            -----
            returns the dataframe with the audio array appended
        '''

        # iterate the dataframe to store the audio array
        for _, data in df.iterrows():
            data['file'] = data['file'].replace(self.root_path_to_be_removed, self.root_path_to_be_replaced)
            data.audio['path'] = data.audio['path'].replace(self.root_path_to_be_removed, self.root_path_to_be_replaced)
            data.audio['array'] = librosa.load(data.audio['path'], sr=16000)[0]

        return df


    def load_audio_dataset(self) -> DatasetDict:
        '''
            to read the train, dev and test pickle file and form the final transformers DatasetDict
        '''

        with open(self.train_pkl_dir, 'rb') as f:
            df_train = pickle.load(f)

        with open(self.dev_pkl_dir, 'rb') as f:
            df_dev = pickle.load(f)

        with open(self.test_pkl_dir, 'rb') as f:
            df_test = pickle.load(f)

        # loads the audio tensor from the audio filepath
        df_train = self.load_audio_tensor_to_df(df_train)
        df_dev = self.load_audio_tensor_to_df(df_dev)
        df_test = self.load_audio_tensor_to_df(df_test)

        # make it into a DatasetDict Object
        dataset = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "dev": Dataset.from_pandas(df_dev),
            "test": Dataset.from_pandas(df_test)
        })

        return dataset

    
    def prepare_dataset(self, batch):
        '''
            to prepare the final dataset that is to be fed into the pretrained whisper model for finetuning, this method is used in the huggingface dataset.map(...) call
        '''
        
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = self.tokenizer(batch["text"]).input_ids
        
        return batch

    
    def compute_metrics(self, pred):
        '''
            to evaluate the wer of the model on the dataset
        '''
        
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        get_wer = WER(predictions=pred_str, references=label_str)
        #wer = 100 * get_wer.compute()

        wer = get_wer.compute()

        # wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    
    def train_trainer_ddp(self) -> None:
        '''
            set up the trainer ddp
        '''

        # load the dataset
        dataset = self.load_audio_dataset()

        # preprocess the dataset to get only the input features and the labels based on the annotation and the audio array passed in
        dataset = dataset.map(self.prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)

        # initialise the processor here
        processor = WhisperProcessor.from_pretrained(self.pretrained_whisper_model_dir, language=self.finetuned_language, task="transcribe")

        # initialise collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        # get the model
        model = WhisperForConditionalGeneration.from_pretrained(self.pretrained_whisper_model_dir)

        # override generation arguments - no tokens are forced as decoder outputs
        model.config.use_cache = False
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        for name, param in model.named_parameters():
            param.requires_grad = False
            if name == 'model.decoder.layer_norm.weight' or name == 'model.decoder.layer_norm.bias' or name.startswith('model.decoder.layers.11.'):
                param.requires_grad = True
            # print(name, param.requires_grad)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.finetuned_output_dir,  # change to a repo name of your choice
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            group_by_length=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
            gradient_checkpointing=True,
            #max_steps=200,
            fp16=True,
            evaluation_strategy="steps",
            predict_with_generate=True,
            generation_max_length=400, #225
            num_train_epochs=self.num_train_epochs,
            save_steps=self.save_eval_logging_steps,
            eval_steps=self.save_eval_logging_steps,
            logging_steps=self.save_eval_logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            save_total_limit=1,
            greater_is_better=False,
            push_to_hub=False,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=processor.feature_extractor,
            callbacks=[TensorBoardCallback(),],
        )

        processor.save_pretrained(training_args.output_dir)
    
        trainer.train()

    def __call__(self) -> None:
        '''
            call the train_trainer_ddp method in the distributed way
        '''

        #return self.train_trainer_ddp()
        return accelerate.notebook_launcher(self.train_trainer_ddp, args=(), num_processes=self.num_processes)
        

if __name__ == '__main__':
    w = WhisperFinetuning(
            train_pkl_dir='/whisper_finetuning/datasets/jtubespeech/ms_2/annotated_data_whisper_ms/train_small.pkl', 
            dev_pkl_dir='/whisper_finetuning/datasets/jtubespeech/ms_2/annotated_data_whisper_ms/dev.pkl', 
            test_pkl_dir='/whisper_finetuning/datasets/jtubespeech/ms_2/annotated_data_whisper_ms/test.pkl', 
            root_path_to_be_removed='/stt_with_kenlm_pipeline', 
            root_path_to_be_replaced='/whisper_finetuning',
            pretrained_whisper_model_dir='/whisper_finetuning/models/whisper/whisper-small',
            finetuned_language='malay',
            finetuned_output_dir='/whisper/models/whisper/whisper-small-malay',
            learning_rate=1e-5,
            weight_decay=5e-5,
            warmup_steps=100,
            num_train_epochs=2,
            save_eval_logging_steps=300,
            num_processes=1
        )

    w()