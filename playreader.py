#!/usr/bin/env python3
# coding: utf-8


"""
Load a play from Project Gutenberg and analyze the speech acts of its figures.


Usage:
    playreader.py  (--url=<url> | --infile=<fname>) [--save-file=<fname>]


Options:
    -h --help              Show this screen.
    --url=<fname>          Load a remote file.
    --infile=<fname>       Load a local file.
    --save-file=<fname>    Save a remote file locally after processing.

Example:
    python playreader.py --url http://www.gutenberg.org/ebooks/9108.txt.utf-8 --save-file lessing_galotti.txt
    python playreader.py --infile lessing_galotti.txt
"""

from pathlib import Path
from collections import defaultdict
import requests
import re
from collections import Counter
from docopt import docopt

from pyvis.network import Network
import networkx as nx
import pandas as pd


class _defaultdict(defaultdict):
    def __add__(self, other):
        return other


def CountTree():
    return _defaultdict(CountTree)


def extract_figures(text, thres=2):

    figures = re.findall("\n\n(.+?)(?: \(.+\))?\.", text, re.M)
    figures = Counter(figures)
    figures = [fig for fig, n in figures.items() if n >= thres]
    figures = [fig for fig in figures if 'szene' not in fig.lower()]

    return figures


def extract_acts(text):
    acts = re.findall("\n\n(.*Aufzug*|.*act.*)\n\n", text, re.M)

    return acts


def remove_gutenburg_headers(book_text):
    book_text = book_text.replace('\r', '')
    start_match = re.search(r'\*{3}\s?START.+?\*{3}', book_text)
    end_match = re.search(r'\*{3}\s?END.+?\*{3}', book_text)
    try:
        book_text = book_text[start_match.span()[1]:end_match.span()[0]]
    except AttributeError:
        print('No match found')

    book_text = re.sub("This book content .*", "", book_text, re.IGNORECASE)
    book_text = re.sub(".*gutenberg.*", "", book_text, re.IGNORECASE)

    return book_text


class PlayReader():
    def __init__(self):
        self.text = ''

    def load_from_url(self, url):
        # make a request and get a response object
        response = requests.get(url)

        # get the source from the response object
        self.text = response.text

    def load_from_file(self, fname):
        with open(fname, mode="r", encoding="utf-8") as f:
            self.text = f.read()

    def write_to_file(self, fname):
        with open(fname, mode="w", encoding="utf-8") as f:
            f.write(self.text)

    def parse_book(self, ):
        text_clean = remove_gutenburg_headers(self.text)
        self.figures = extract_figures(text_clean)

        self.acts = extract_acts(text_clean)
        self.text_clean = text_clean

    def collect_stats(self):

        g = nx.Graph()

        # stats['act']['figure']['text'] = 1
        stats = CountTree()

        current_figure = ''
        current_act = ''

        for line in self.text.split('\n'):

            if not line.strip():
                continue

            candidate_figure = [fig for fig in self.figures if line.startswith(fig)]
            candidate_act = [act for act in self.acts if line.startswith(act)]

            if candidate_figure:
                candidate_figure = candidate_figure[0]

                if g.has_edge(current_figure, candidate_figure):
                    g[current_figure][candidate_figure]['weight'] += 1
                elif current_figure in self.figures and candidate_figure in self.figures:
                    g.add_edge(current_figure, candidate_figure, weight=1)

                current_figure = candidate_figure

            if candidate_act:
                current_act = candidate_act[0]
                current_figure = ''  # reset figure
                continue

            if current_act:
                stats[current_figure][current_act]['text'] += 1

        self.stats = stats
        self.graph = g

    def plot_graph(self):
        g = self.graph
        g.remove_edges_from(nx.selfloop_edges(g))

        degree = dict(g.degree)
        weight = nx.get_edge_attributes(g, "weight")
        nx.set_node_attributes(g, degree, "size")
        nx.set_edge_attributes(g,  weight, 'value')

        nt = Network('500px', '500px',)  # notebook=True

        # nt.show_buttons()
        # populates the nodes and edges data structures
        nt.from_nx(g)
        nt.show('nx.html')

    def write_stats(self):
        rows = []

        try:
            rows = [{'figure': fig, "act": act, "n_lines": self.stats[fig][act]['text']}
                    for fig in self.figures for act in self.acts]
        except ValueError:
            pass

        df = pd.DataFrame(rows)
        df["n_lines"][df.n_lines == {}] = 0
        df.to_csv('lines_per_figure_act.csv',  index=False)

        df_tot = df[['figure', 'n_lines']].groupby('figure').sum().reset_index()
        df_tot = df_tot.sort_values("n_lines", ascending=False)
        df.to_csv('total_lines_per_figure.csv', index=False)


if __name__ == "__main__":

    args = docopt(__doc__)
    url = args["--url"]
    f_in = Path(args["--infile"]) if args["--infile"] else None
    f_out = Path(args["--save-file"]) if args["--save-file"] else None

    book = PlayReader()
    if url:
        book.load_from_url(url)
    else:
        book.load_from_file(f_in)

    book.parse_book()
    book.collect_stats()
    book.plot_graph()
    book.write_stats()

    if f_out:
        book.write_to_file(f_out)
