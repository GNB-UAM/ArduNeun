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
    infile = sys.argv[1] if len(sys.argv) > 1 else 'sizes.csv'
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

    import re
    import numpy as np
    import matplotlib.pyplot as plt

    with open('sizes.csv') as f:
        rows = list(csv.DictReader(f))

    # "C1: ESP32-s3_Basic w/o Neun" -> caso "C1: Basic w/o Neun", placa "ESP32-S3"
    LABEL_RE = re.compile(r'^(?P<case>C\d+):\s*(?P<board>[^_]+)_(?P<name>.*)$')

    cases = {}  # etiqueta del caso -> {placa: flash usada en bytes}
    for r in rows:
        m = LABEL_RE.match(r['label'])
        if not m:
            print(f"Etiqueta no reconocida: {r['label']!r}")
            continue
        case_label = f"{m['case']}: {m['name']}"
        board = m['board'].upper()                      # ESP32-S3 / ESP8266
        cases.setdefault(case_label, {})[board] = int(r['text']) + int(r['data'])

    board_colors = {
        'ESP32-S3': 'steelblue',
        'ESP8266':  'darkorange',
    }

    # Líneas de capacidad: (texto leyenda, bytes, placa/color)
    capacities = [
        ('ESP32-S3 flash (4 MB)', 4 * 1024 * 1024, 'ESP32-S3'),
        ('ESP8266 flash (4 MB)',  4 * 1024 * 1024, 'ESP8266'),
        ('ESP8266 flash (1 MB)',  1 * 1024 * 1024, 'ESP8266'),
    ]

    plt.figure(figsize=(10, 5.5))

    # Barras agrupadas: una por placa dentro de cada caso
    x = np.arange(len(cases))
    width = 0.38
    for i, (board, color) in enumerate(board_colors.items()):
        values = [cases[c].get(board, 0) for c in cases]
        plt.bar(x + (i - 0.5) * width, values, width,
                color=color, label=board, zorder=3)

    # Líneas de referencia (mismo color que la placa). Si comparten
    # capacidad, se desplaza el patrón de guiones para que se vean ambas.
    dash = 6
    seen = {}
    for name, capacity, board in capacities:
        n = seen.get(capacity, 0)
        seen[capacity] = n + 1
        plt.axhline(y=capacity, color=board_colors[board],
                    linestyle=(n * dash, (dash, dash)),
                    linewidth=1.5, label=name, zorder=2)

    kb_ticks = [0, 256 * 1024, 512 * 1024, 768 * 1024]
    mb_ticks = [i * 1024 * 1024 for i in range(1, 5)]
    tick_labels = ['0 KB', '256 KB', '512 KB', '768 KB'] + [f'{i} MB' for i in range(1, 5)]

    plt.yticks(kb_ticks + mb_ticks, tick_labels)
    plt.xticks(x, list(cases.keys()), rotation=20, ha='right')
    plt.ylabel('Flash used')
    plt.title('Flash memory usage per case')
    plt.grid(axis='y', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
    plt.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('flash_usage.pdf', format='pdf')
    plt.show()