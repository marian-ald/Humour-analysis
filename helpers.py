import os
import sys
import csv
import json
import pickle
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.spellcorrect import SpellCorrector
import emoji
import string
from langdetect import detect
import re


import pandas as pd
import spacy
import en_core_web_sm
from normalise import normalise
import nltk

nltk.download('brown')

for dependency in ("brown", "names", "wordnet", "averaged_perceptron_tagger", "universal_tagset"):
    nltk.download(dependency)

class data_process:
    def __init__(self):
        self.root_dir = "CrisisLexT26/"
        self.count = 0

    """
        Load headlines info and replace the edit word in the original headline, resulting the
        edited headline. Write the text and the score in a new csv file
    """
    def save_generate_edited(self, csv_file, new_fname, d_set):

        pd_file = pd.read_csv(csv_file)
        # id,original,edit,grades,meanGrade

        prep_edited_text = []
        prep_original = []

        for i in range(len(pd_file["id"])):

            # Replace the marked <word/> in the original sentence with the edit
            to_replace = '[-—|\[\]!?()“”‘/,’…\'.<>:";]'
            new_headline = re.sub(r"<(.*?)/>", pd_file['edit'][i], pd_file['original'][i])
            new_headline = re.sub(to_replace, ' ', new_headline)
            new_original = re.sub(to_replace, ' ', pd_file['original'][i])

            prep_edited_text.append(new_headline)
            prep_original.append(new_original)

        # Build a dictionary with the new fields
        if d_set == 'train':
            new_dict = {'id':pd_file['id'], 'original':prep_original, 'edited':prep_edited_text, 'meanGrade':pd_file['meanGrade']}
        else:
            new_dict = {'id':pd_file['id'], 'original':prep_original, 'edited':prep_edited_text, 'meanGrade':[0.9355] * len(prep_original)}

        df = pd.DataFrame(new_dict)
        # Save the dataframe 
        df.to_csv(new_fname)

    """
        Load headlines info and replace the edit word in the original headline, resulting the
        edited headline. Write the text and the score in a new csv file
    """
    def generate_edited(self, csv_file):

        pd_file = pd.read_csv(csv_file)

        #prep_edited_text = []
        original = []
        to_replace = '[/<>]'

        for i in range(len(pd_file["id"])):

            # Replace the marked <word/> in the original sentence with the edit
            #to_replace = '[-—|\[\]!?()“”‘/,’…\'.<>:";]'
            #new_headline = re.sub(r"<(.*?)/>", pd_file['edit'][i], pd_file['original'][i])
#           new_headline = re.sub(to_replace, '', new_headline)
            new_original = re.sub(to_replace, '', pd_file['original'][i])

            original.append(new_original)

        return original

    """
        Load original/edited/grade and preprocess the text.
        Write back to another file the results
        d_set flag can be 'train' or 'dev'
    """
    def load_edited_csv(self, csv_file, new_fname, d_set, prep):
        pd_file = pd.read_csv(csv_file)

        if prep == True:
            # Preprocess the headlines
            prep_original = self.preprocess(pd_file['original'])
            prep_edited = self.preprocess(pd_file['edited'])
 
            # Write de preprocessed headlines to a new file
            new_dict = {'original':prep_original, 'edited':prep_edited, 'meanGrade':pd_file['meanGrade']}

            df = pd.DataFrame(new_dict)
            # Save the dataframe 
            df.to_csv(new_fname)

        return pd_file['original'].tolist(), pd_file['edited'].tolist(), pd_file['meanGrade'].tolist()


    """
        Load original/edited/grade = preprocessed text.
    """
    def load_edited_prep_csv(self, csv_file):
        pd_file = pd.read_csv(csv_file)

        return pd_file['original'].tolist(), pd_file['edited'].tolist(), pd_file['meanGrade'].tolist()

    """
        Return the hlines ids for the test dataset
    """
    def load_dev_ids(self, csv_file):
        pd_file = pd.read_csv(csv_file)
        return pd_file['id'].tolist()

    def build_dev_results(self, ids, preds, f_name):
        new_dict = {'id':ids, 'pred':preds}

        df = pd.DataFrame(new_dict)
        df.to_csv(f_name)
        print("Saved the predicted values into: " + f_name) 



    # Return concatenated titles splited by 'separatorSeparator'
    def build_text(self, csv_file, d_set):
        orig_hline, edit_hline, grade = self.load_edited_csv(csv_file, None, d_set, False)

        do_count = False
        min_len = 100
        total_words = 0
        max_len = 0
        if do_count == True:
            for i in range(len(orig_hline)):
                spl = orig_hline[i].split()
                total_words = total_words + len(spl)
                if len(spl) > max_len:
                    max_len = len(spl)
                if len(spl) < min_len:
                    min_len = len(spl)


            mean = total_words / len(orig_hline)
            print("mean" + str(mean))
            print("max len ")
            print(max_len)
            print("min len ")
            print(min_len)
            sys.exit()

        combined_hline = [orig + " separatorSeparator " + edited for orig, edited in zip(orig_hline, edit_hline)]
        return combined_hline, grade

    def build_text2(self, csv_file, d_set):
        orig_hline, edit_hline, grade = self.load_edited_csv(csv_file, d_set)

        return orig_hline, edit_hline, grade
    
    def save_pickle(self, file_name, obj_name):
        pickle_out = open(file_name, "wb")
        pickle.dump(obj_name, pickle_out)
        pickle_out.close()


    def open_pickle(self, file_name):
        pickle_in = open(file_name, "rb")
        obj = pickle.load(pickle_in)
        pickle_in.close()
        return obj


    def hardc_string(self, s):
        s = re.sub(r'((Barack )?Obama )', " [PRESIDENT1] ", s)
        s = re.sub(r'((Donald )?Trump )', " [PRESIDENT2] ", s)
        s = re.sub(r'((Vladimir )?Putin )', " [PRESIDENT3] ", s)
        s = re.sub(r'(PM )', "prime minister ", s)

        return s
        
    """
        Preprocess headlines
    """
    def preprocess(self, h_lines):
        prep_h_lines = []
        save_to_file = []
        
        nlp = en_core_web_sm.load()

        for i in range(len(h_lines)):
            doc = nlp(h_lines[i])
            sentence = doc.text

            # Delete [] brackets
            #sentence = re.sub('[\[\]]', '', sentence)
            #sentence = re.sub('[-—|\[\]!?()“”,’…\'.:";]', ' ', sentence)

            sentence = self.hardc_string(sentence)

            # Normalise the abbreviations
            #list_words = sentence.split()
            #list_words = normalise(list_words, verbose=False)
            #sentence = " ".join(list_words)
            
            for ent in doc.ents:
                sentence = re.sub(ent.text, '[' + ent.label_ + ']', sentence)
            

            # Transform all letters in lowercase
            sentence = sentence.lower()

            # Make capital letters inside [...]
            sentence = re.sub(r'(\[[a-z]*\])', lambda pat: pat.group(0).upper(), sentence)

            # Remove [] brackets
            sentence = re.sub('[\[\]]', ' ', sentence)

            #line = re.sub('[-—|\[\]!?()“”,’…\'.:";]', '', line)

            prep = sentence
            
            prep_h_lines.append(prep)

        new_dict = {'text':save_to_file}
        df = pd.DataFrame(new_dict)
        print("Wrote prep headlines in prep_h_lines.csv")
        df.to_csv("prep_h_lines_lower.csv") 


        return prep_h_lines

    """
        Display first n natural tweets from 'list'
    """
    def to_string(self, n, obj_list):
        if len(obj_list) == 0:
            print("Empty lists in to_string method")
            return

        for i in range(n):
            print(obj_list[i])

    """
        Write a column with meanGrade prediction in the dev.csv file
    """
    def write_meanGrade_col(self, csv_file, meaGrade_col):
        pd_file = pd.read_csv(csv_file)
        pd_file['meanGrade'] = meanGrade_col
        pd_file.to_csv("../data/task-1/dev_meanGrade.csv")

        

#d = data_process()
#d.write_meanGrade_col("../data/task-1/dev.csv")
#h_lines = d.generate_edited("../data/task-1/train.csv")
#d.preprocess(h_lines)


################# Steps until train
#d.save_generate_edited("../data/task-1/train.csv", "../data/task-1/edited_train.csv", 'train')
#d.save_generate_edited("../data/task-1/dev.csv", "../data/task-1/edited_dev.csv", 'dev')

#d.load_edited_csv("../data/task-1/edited_train.csv", "../data/task-1/prep_train.csv",'train', True)
#d.load_edited_csv("../data/task-1/edited_dev.csv", "../data/task-1/prep_dev.csv",'dev', True)


