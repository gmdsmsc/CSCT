import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == '__main__':
    _thisdir = Path(__file__).parent.absolute()
    file_path_str = _thisdir / 'create_data' / 'fixed_gravity' / "positions_141.txt"

    with open(file_path_str, 'r') as f:
        lines = f.readlines()  
    positions = [float(line.strip()) for line in lines]
    df = pd.DataFrame(positions, columns=['position'])

    sns.lineplot(data=df, x=df.index, y='position').set(title='Position over Time')
    plt.show()
