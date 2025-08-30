import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path_str = "E:/python_home/CSCT/benchmarking/create_data/short_window/positions_141.txt"
    with open(file_path_str, 'r') as f:
        lines = f.readlines()  
    positions = [float(line.strip()) for line in lines]
    df = pd.DataFrame(positions, columns=['position'])

    sns.lineplot(data=df, x=df.index, y='position').set(title='Position over Time')
    plt.show()
