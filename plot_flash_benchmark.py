#!/usr/bin/env python3
"""
Parse a build-log file containing repeated blocks like:

    Basic code without Neun
    Calculating size .pio/build/esp32-s3-flash-empty/firmware.elf
      text    data     bss     dec     hex filename
    201403   89284  576421  867108   d3b24 .pio/build/esp32-s3-flash-emp
    ty/firmware.elf
    One neuron type
    Calculating size ...
      text    data     bss     dec     hex filename
    ... numbers ...

There can be any number (N) of these blocks. The path after
"Calculating size" sometimes wraps onto the next line (as shown above);
this script handles that by matching across newlines and stripping
whitespace back out.

Usage:
    python parse_sizes.py path/to/logfile.txt
    (defaults to "sizes.txt" if no argument given)

Produces:
    - printed list of parsed entries
    - sizes.csv with the same data
"""

import re
import csv
import sys

# Matches "Calculating size <path>" (path may wrap across a line break),
# followed by the header row, followed by the 5 numeric columns.
# We don't need to capture the trailing "filename" column since it is
# always identical to <path>.
BLOCK_RE = re.compile(
    r'Calculating size\s*(?P<path>.*?)\s*\n'
    r'\s*text\s+data\s+bss\s+dec\s+hex\s+filename\s*\n'
    r'\s*(?P<text>\d+)\s+(?P<data>\d+)\s+(?P<bss>\d+)\s+(?P<dec>\d+)\s+(?P<hexv>[0-9a-fA-F]+)\b',
    re.DOTALL,
)


def clean_path(p: str) -> str:
    """Remove any embedded whitespace/newlines caused by line-wrapping."""
    return re.sub(r'\s+', '', p)


def parse_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    matches = list(BLOCK_RE.finditer(content))
    results = []
    prev_end = 0

    for i, m in enumerate(matches):
        # The "label" (e.g. "Basic code without Neun", "One neuron type")
        # is the last non-blank line of text sitting between the end of
        # the previous block and the start of this "Calculating size".
        label_chunk = content[prev_end:m.start()].strip()
        label = label_chunk.splitlines()[-1].strip() if label_chunk else f'block_{i}'

        results.append({
            'label': label,
            'path': clean_path(m.group('path')),
            'text': int(m.group('text')),
            'data': int(m.group('data')),
            'bss': int(m.group('bss')),
            'dec': int(m.group('dec')),
            'hex': m.group('hexv'),
        })
        prev_end = m.end()

    return results


if __name__ == '__main__':
    infile = sys.argv[1] if len(sys.argv) > 1 else 'sizes.txt'
    entries = parse_file(infile)

    if not entries:
        print(f'No "Calculating size" blocks found in {infile!r}.')
        sys.exit(0)

    for e in entries:
        print(e)

    out_csv = 'sizes.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=entries[0].keys())
        writer.writeheader()
        writer.writerows(entries)

    print(f'\nParsed {len(entries)} block(s) -> {out_csv}')

    import csv
    import matplotlib.pyplot as plt

    with open('sizes.csv') as f:
        rows = list(csv.DictReader(f))

    labels = [r['label'] for r in rows]
    flash = [int(r['text']) + int(r['data']) for r in rows]  # bytes written to flash

    # Common board flash sizes, in bytes (adjust to match your actual hardware/partition table)
    board_capacities = {
        'ESP32-S3 (8MB flash)':  8 * 1024 * 1024,
        'ESP32-S3 (4MB flash)':  4 * 1024 * 1024,
        'ESP8266 (4MB flash)':   4 * 1024 * 1024,
        'ESP8266 (1MB flash)':   1 * 1024 * 1024,
    }

    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(labels, flash, color='steelblue', zorder=3)

    # Draw a horizontal reference line for each board capacity
    colors = ['crimson', 'darkorange', 'seagreen', 'purple']
    for (board, capacity), color in zip(board_capacities.items(), colors):
        plt.axhline(y=capacity, color=color, linestyle='--', linewidth=1.5,
                    label=f'{board} max ({capacity / 1024 / 1024:.0f} MB)', zorder=2)

    plt.ylabel('Flash used (bytes)')
    plt.title('Flash memory usage per case')
    plt.xticks(rotation=20, ha='right')
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig('flash_usage.png', dpi=150)
    plt.show()