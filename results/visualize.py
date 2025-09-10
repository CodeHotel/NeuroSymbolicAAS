#!/usr/bin/env python3
"""
"C:/Program Files/Python312/python.exe" visualize.py trace.xlsx -o timeline.png     
Timeline saved to timeline.png

visualize.py: Plot a Gantt-style timeline of operations executed by machines M1, M2, M3

Usage:
    python visualize.py trace.xlsx -o timeline.png

Requires:
    pandas, matplotlib, openpyxl
Install with:
    pip install pandas matplotlib openpyxl
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Visualize machine operations timeline.')
    parser.add_argument('input', help='Path to trace Excel file')
    parser.add_argument('-o', '--output',
                        help='Path to save output figure',
                        default='timeline.png')
    args = parser.parse_args()

    # Load trace data
    df = pd.read_excel(args.input)

    # Extract start and end times for each operation
    df_start = df[df['event'] == 'start'][['part', 'job', 'operation', 'machine', 'time']]
    df_start = df_start.rename(columns={'time': 'start'})
    df_end = df[df['event'] == 'end'][['part', 'job', 'operation', 'time']]
    df_end = df_end.rename(columns={'time': 'end'})

    # Merge to get intervals
    df_ops = pd.merge(df_start, df_end,
                      on=['part', 'job', 'operation'],
                      how='inner')

    # Determine machine order
    machines = sorted(df_ops['machine'].unique())
    machine_to_y = {m: i for i, m in enumerate(machines)}

    # Plot
    fig, ax = plt.subplots()
    for _, row in df_ops.iterrows():
        y = machine_to_y[row['machine']]
        start = row['start']
        end = row['end']
        duration = end - start
        ax.barh(y, duration, left=start)
        ax.text(start + duration / 2, y, row['operation'],
                va='center', ha='center')

    # Formatting
    ax.set_yticks(list(machine_to_y.values()))
    ax.set_yticklabels(list(machine_to_y.keys()))
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Machine Operation Timeline')
    ax.set_xlim(left=0)
    plt.tight_layout()

    # Save & show
    fig.savefig(args.output)
    print(f"Timeline saved to {args.output}")
    plt.show()

if __name__ == '__main__':
    main()