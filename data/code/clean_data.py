import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### DEFINE
def main():
    df = pd.read_csv('output/data_merged.csv')
    plot_data(df)
    df = clean_data(df)
    df.to_csv('output/data_cleaned.csv', index = False)

def plot_data(df):
    fig, ax = plt.subplots()
    ax.hist(df['chips_sold'])
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y / len(df) * 100:.0f}%'))
    ax.set_ylabel('Percentage')

    plt.savefig('output/chips_sold.pdf')

def clean_data(df):
    df['chips_sold'][df['chips_sold'] == -999999] = np.NaN
    return(df)
    
### EXECUTE
main()
