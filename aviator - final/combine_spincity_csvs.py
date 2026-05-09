import csv
import glob
import os

files = sorted(glob.glob('spincity*.csv'))
print('Found', len(files), 'spincity CSV files')
all_headers = []
rows = []
for f in files:
    with open(f, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            print('Skipping empty file', f)
            continue
        header = [h.strip() for h in header]
        lower = [h.lower() for h in header]
        if 'multiplier' not in lower:
            header = ['multiplier'] + header[1:]
        if not all_headers:
            all_headers = header[:]
        else:
            for h in header:
                if h not in all_headers:
                    all_headers.append(h)
        for row in reader:
            if not row:
                continue
            row = [cell.strip() for cell in row]
            rows.append((header, row))

combined_path = 'compiled_spincity_data.csv'
with open(combined_path, 'w', newline='', encoding='utf-8') as out:
    writer = csv.writer(out)
    writer.writerow(all_headers)
    for header, row in rows:
        outrow = [''] * len(all_headers)
        for i, h in enumerate(header):
            if i >= len(row):
                continue
            try:
                j = all_headers.index(h)
            except ValueError:
                continue
            outrow[j] = row[i]
        writer.writerow(outrow)

multiplier_count = 0
with open(combined_path, newline='', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        if row.get('multiplier', '').strip() != '':
            multiplier_count += 1

print('Wrote combined file:', combined_path)
print('Total multiplier rows:', multiplier_count)
