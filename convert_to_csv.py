import csv
import sys

"""Simple script to convert whitespace or comma separated text file to CSV.
Usage: python convert_to_csv.py input.txt output.csv
"""

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_to_csv.py input.txt output.csv")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]

    with open(infile, 'r', newline='') as fin, open(outfile, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        for row in reader:
            # skip empty rows
            if not row:
                continue
            writer.writerow(row)

    print(f"Converted {infile} to {outfile}")


if __name__ == '__main__':
    main()