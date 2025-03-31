import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot a CSV track.")
    parser.add_argument('--csv', type=str, default='test.csv',
                        help='Path to the CSV file with track data.')
    args = parser.parse_args()
    
    # 2. Attempt to read CSV
    filepath = args.csv
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return
    
    # 3. Define colors for tags
    colors = {
        'yellow': 'gold',
        'blue': 'royalblue',
        'orange': 'orange',
        'none': 'grey'
    }
    
    # 4. Plot the track points
    plt.figure(figsize=(12, 8))
    
    for tag in data['tag'].unique():
        subset = data[data['tag'] == tag]
        plt.scatter(
            subset['x'], subset['y'],
            label=tag,
            color=colors.get(tag, 'black'),
            s=20
        )
    
    # 5. Additional plot styling
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Formula Student Track')
    plt.legend()
    plt.axis('equal')  # Equal aspect ratio for correct proportions
    plt.grid(True)
    plt.tight_layout()
    
    # 6. Show plot
    plt.show()

if __name__ == "__main__":
    main()
