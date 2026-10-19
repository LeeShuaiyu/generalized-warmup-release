"""Compact tables for publication notebooks."""
import html
from IPython.display import HTML, display

def folded_table(title, frame):
    display(HTML('<details><summary>' + html.escape(title) + '</summary><div style="overflow:auto;max-height:620px">' + frame.to_html(border=0, na_rep='—', float_format=lambda v: f'{v:.6g}') + '</div></details>'))

def folded_comparison(frame):
    columns = [c for c in frame.columns if not str(c).lower().startswith(('match', 'agreement'))]
    folded_table('Reference values', frame.loc[:, columns])

def compact_settings(frame):
    return frame.reset_index().drop(columns=['Seed'], errors='ignore').drop_duplicates().reset_index(drop=True)
