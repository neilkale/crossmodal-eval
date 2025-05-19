from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets 
from tqdm import tqdm
from utils.data_utils import CAT_SHORT2LONG 
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_ensembling_ratio', type=float, default=1)
    args = parser.parse_args()

    # run for each subject
    sub_dataset_list = []
    for subject in tqdm(CAT_SHORT2LONG.values(), desc='loading dataset'):
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    avg_text_lengths = []
    avg_image_areas = []
    for sub_dataset in tqdm(sub_dataset_list, desc='measuring dataset size'):
        avg_text_length = 0
        avg_image_area = 0
        
        for i in range(sub_dataset.num_rows):
            avg_text_length += len(sub_dataset['question'][i])
            image_width, image_height = sub_dataset['image_1'][i].size
            avg_image_area += image_width * image_height
        
        avg_text_length /= sub_dataset.num_rows
        avg_image_area /= sub_dataset.num_rows

        avg_text_lengths.append(avg_text_length)
        avg_image_areas.append(avg_image_area)
    
    df = pd.DataFrame()
    df['subject'] = CAT_SHORT2LONG.values()
    df['avg_text_length'] = avg_text_lengths
    df['avg_image_area'] = avg_image_areas
    df.to_csv('avg_text_image_by_subtask.csv', index=False)

def pretty_print(csv_file):
    df = pd.read_csv(csv_file)
    df['avg_text_length'] = df['avg_text_length'].round(2)
    df['avg_image_area'] = df['avg_image_area'].round(2)
    df['ratio'] = df['avg_image_area'] / df['avg_text_length']
    df['ratio'] = df['ratio'].round(2)
    df = df.sort_values(by='avg_image_area', ascending=False)
    print(df.to_string(index=False))

if __name__ == '__main__':
    # main()
    pretty_print('avg_text_image_by_subtask.csv')