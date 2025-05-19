from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from PIL import Image

import textstat
from skimage.measure import shannon_entropy

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def process_subtask(task_name):
    # Load the sub-dataset for the given task (using the 'val' split)
    sub_dataset = load_dataset('BLINK-Benchmark/BLINK', task_name)['val']
    
    # Define a function to compute metrics on a batch of examples.
    def compute_metrics(batch):
        # Compute text lengths for the prompt column
        prompt_lens = [len(text) for text in batch['prompt']]
        # Compute Flesch-Kincaid readability scores for the prompt column
        fk_scores = [textstat.flesch_kincaid_grade(text) for text in batch['prompt']]
        
        # Compute total image area and average Shannon entropy for each example over the image columns.
        image_area_sum = []
        shannon_scores = []
        for i in range(len(batch['prompt'])):
            area = 0
            total_entropy = 0
            valid_images = 0
            for col in ['image_1', 'image_2', 'image_3', 'image_4']:
                image = batch[col][i]
                if image is not None:
                    w, h = image.size
                    area += w * h
                    total_entropy += shannon_entropy(image)
                    valid_images += 1
            image_area_sum.append(area)
            # If images are present, average the entropy; otherwise, return 0.
            shannon_scores.append(total_entropy / valid_images if valid_images > 0 else 0)
        
        return {
            'prompt_len': prompt_lens,
            'fk_score': fk_scores,
            'image_area_sum': image_area_sum,
            'shannon_entropy': shannon_scores
        }

    # Apply the transformation to all examples in batches.
    sub_dataset = sub_dataset.map(compute_metrics, batched=True, batch_size=100, num_proc=4)
    
    # Compute average values by summing the computed fields over all examples.
    avg_text_length = sum(sub_dataset['prompt_len']) / len(sub_dataset)
    avg_image_area = sum(sub_dataset['image_area_sum']) / len(sub_dataset)
    avg_fk_score = sum(sub_dataset['fk_score']) / len(sub_dataset)
    avg_shannon = sum(sub_dataset['shannon_entropy']) / len(sub_dataset)
    
    return avg_text_length, avg_image_area, avg_fk_score, avg_shannon

def main():
    # List of subtasks to process
    subtasks = [
        'Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning',
        'Relative_Reflectance', 'Visual_Correspondence', 'Counting',
        'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity',
        'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation'
    ]

    attention_manip_performance = {'Art_Style': -0.01688835099293241, 'Functional_Correspondence': 0.04342718826754002, 'Multi-view_Reasoning': 0.040881936446855144, 'Relative_Reflectance': -0.005525357939594115, 'Visual_Correspondence': -0.017433789266381925, 'Counting': 0.00019281196976704315, 'IQ_Test': 0.061607280579978385, 'Object_Localization': 0.02294778525424602, 'Semantic_Correspondence': 0.04311220158533051, 'Visual_Similarity': -0.003427768351414986, 'Forensic_Detection': -0.015530127746694183, 'Jigsaw': 0.0008329477093934829, 'Relative_Depth': 0.027429057634607592, 'Spatial_Relation': -0.011261297702760098, 'Overall': 0.012168894103424387}
    
    results = []
    # Process the subtasks concurrently.
    with ProcessPoolExecutor() as executor:
        # Map each task to the process_subtask function.
        for task_name, (avg_text_length, avg_image_area, avg_fk_score, avg_shannon) in zip(
            tqdm(subtasks, desc='Subtasks'),
            executor.map(process_subtask, subtasks)
        ):
            results.append((task_name, avg_text_length, avg_image_area, avg_fk_score, avg_shannon, attention_manip_performance[task_name]))
        
    # Create a DataFrame from the results and save to CSV.
    df = pd.DataFrame(results, columns=['subject', 'avg_text_length', 'avg_image_area', 'avg_fk_score', 'avg_shannon_entropy', 'attention_manip_performance'])
    df.to_csv('avg_text_image_by_subtask.csv', index=False)

def pretty_print(csv_file):
    df = pd.read_csv(csv_file)
    df['attention_manip_performance'] = df['attention_manip_performance'].round(3)
    df['avg_text_length'] = df['avg_text_length'].round(2)
    df['avg_image_area'] = df['avg_image_area'].round(2)
    df['avg_fk_score'] = df['avg_fk_score'].round(2)
    df['avg_shannon_entropy'] = df['avg_shannon_entropy'].round(2)
    df['ratio'] = df['avg_shannon_entropy'] / df['avg_fk_score']
    df['ratio'] = df['ratio'].round(2)
    df = df.sort_values(by='attention_manip_performance', ascending=False)
    print(df.to_string(index=False))

def statistical_analysis(csv_file):
    # Load the CSV into a DataFrame.
    df = pd.read_csv(csv_file)
    
    # Define the columns to be normalized. You might decide
    # whether to include the dependent variable (attention_manip_performance)
    # or only the predictors. Here we standardize all variables.
    features = ['avg_text_length', 'avg_image_area', 'avg_fk_score', 'avg_shannon_entropy']
    
    # Create scalers for predictors and for the dependent variable separately.
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Normalize the predictors
    df_features = df[features]
    df_features_scaled = pd.DataFrame(scaler_X.fit_transform(df_features), columns=features)
    
    # Normalize the dependent variable
    df['attention_manip_performance_normalized'] = scaler_y.fit_transform(df[['attention_manip_performance']])
    
    # Combine the normalized predictors and dependent variable into one DataFrame.
    df_norm = df_features_scaled.copy()
    df_norm['attention_manip_performance_normalized'] = df['attention_manip_performance_normalized']
    
    # Print a correlation matrix on the normalized variables.
    variables_norm = ['attention_manip_performance_normalized'] + features
    print("Correlation Matrix (Normalized Variables):")
    print(df_norm[variables_norm].corr())
    print("\n")
    
    # Set up the regression. Here we use the normalized dependent variable.
    y = df_norm['attention_manip_performance_normalized']
    X = df_norm[features]
    X = sm.add_constant(X)
    
    # Fit the OLS regression model.
    model = sm.OLS(y, X).fit()
    
    # Print the regression summary.
    print("Regression Analysis Summary (Normalized Variables):")
    print(model.summary())

def plot(x_var = 'avg_text_length', y_var = 'attention_manip_performance'):
    import matplotlib.pyplot as plt
    import numpy as np
    from adjustText import adjust_text
    
    df = pd.read_csv('avg_text_image_by_subtask.csv')
    df[x_var] = df[x_var].round(2)
    df[y_var] = df[y_var].round(2)
    df = df.sort_values(by=y_var, ascending=False)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_var], df[y_var], color='blue', alpha=0.5)
    
    # Add titles and labels
    plt.title(f'Scatter Plot of {y_var} vs {x_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    texts = []
    for idx, row in df.iterrows():
        texts.append(plt.text(row[x_var], row[y_var], str(row['subject']), fontsize=9))

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'))

    # Add line of best fit
    z = np.polyfit(df[x_var], df[y_var], 1)
    p = np.poly1d(z)
    plt.plot(df[x_var], p(df[x_var]), color='red', linewidth=2, label='Line of Best Fit (R= {:.2f})'.format(np.corrcoef(df[x_var], df[y_var])[0, 1]))
    plt.legend()
    
    # Show the plot
    plt.savefig(f'comparing_subtasks_{x_var}_v_{y_var}.png')

if __name__ == '__main__':
    main()
    pretty_print('avg_text_image_by_subtask.csv')
    statistical_analysis('avg_text_image_by_subtask.csv')
    plot(x_var='avg_text_length', y_var='attention_manip_performance')
