import nbformat
import glob
import os

NOTEBOOK_STYLE_CELL = """#
from IPython.display import HTML, display

display(HTML('''
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Space+Mono&display=swap');

  :root {
    --nothing-black:   #0A0A0A;
    --nothing-surface: #111111;
    --nothing-border:  #2A2A2A;
    --nothing-white:   #F5F5F5;
    --nothing-gray-1:  #CCCCCC;
    --nothing-gray-2:  #888888;
    --nothing-gray-3:  #555555;
    --nothing-red:     #FF1C1C;
    --font-ndot:       'Share Tech Mono', monospace;
    --font-body:       'Space Mono', monospace;
  }

  body, .jp-Notebook {
    background: var(--nothing-black) !important;
    font-family: var(--font-body) !important;
  }

  .jp-Cell { border-radius: 0 !important; }

  .jp-InputArea-editor, .jp-CodeCell .jp-Editor {
    background: var(--nothing-surface) !important;
    border: 1px solid var(--nothing-border) !important;
    border-radius: 0 !important;
    font-family: var(--font-body) !important;
    font-size: 12px !important;
  }

  .jp-OutputArea-output {
    background: #0D0D0D !important;
    border-left: 2px solid var(--nothing-border) !important;
    font-family: var(--font-body) !important;
    color: var(--nothing-gray-1) !important;
    font-size: 12px !important;
  }

  .jp-MarkdownOutput h1, .jp-MarkdownOutput h2, .jp-MarkdownOutput h3 {
    font-family: var(--font-ndot) !important;
    color: var(--nothing-white) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--nothing-border) !important;
    padding-bottom: 8px;
  }

  .jp-MarkdownOutput h1 { font-size: 22px !important; }
  .jp-MarkdownOutput h2 { font-size: 16px !important; }
  .jp-MarkdownOutput h3 { font-size: 13px !important; color: var(--nothing-gray-2) !important; }

  .jp-MarkdownOutput p, .jp-MarkdownOutput li {
    color: var(--nothing-gray-2) !important;
    font-family: var(--font-body) !important;
    font-size: 12px !important;
  }

  .jp-MarkdownOutput code {
    background: var(--nothing-surface) !important;
    color: var(--nothing-red) !important;
    border: 1px solid var(--nothing-border) !important;
    border-radius: 0 !important;
    padding: 1px 5px;
    font-size: 11px !important;
  }

  .jp-Toolbar {
    background: #070707 !important;
    border-bottom: 1px solid var(--nothing-border) !important;
  }
</style>
'''))

# And configure matplotlib globally:
import matplotlib as mpl
mpl.rcParams.update({
    'figure.facecolor': '#0A0A0A', 'axes.facecolor': '#111111',
    'axes.edgecolor': '#2A2A2A', 'axes.labelcolor': '#888888',
    'text.color': '#888888', 'xtick.color': '#555555', 'ytick.color': '#555555',
    'grid.color': '#2A2A2A', 'grid.linewidth': 0.5, 'grid.linestyle': ':',
    'axes.grid': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'monospace', 'figure.titlesize': 11,
    'legend.facecolor': '#111111', 'legend.edgecolor': '#2A2A2A',
    'lines.linewidth': 1.5, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.titlecolor': '#F5F5F5',
})

NOTHING_RED    = '#FF1C1C'
NOTHING_WHITE  = '#F5F5F5'
NOTHING_GRAY   = '#888888'
NOTHING_DIM    = '#555555'
NOTHING_MUTED  = '#3A3A3A'
"""

def update_notebooks():
    notebooks = glob.glob("notebooks/*.ipynb")
    for nb_path in notebooks:
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Check if styling cell already exists
            has_style = any("NOTHING OS — Notebook Styling" in cell.source for cell in nb.cells)
            
            if not has_style:
                style_cell = nbformat.v4.new_code_cell(source=NOTEBOOK_STYLE_CELL)
                nb.cells.insert(0, style_cell)
                
                with open(nb_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                print(f"Updated {nb_path} with Nothing OS styling.")
            else:
                print(f"{nb_path} already has Nothing OS styling.")
        except Exception as e:
            print(f"Failed to update {nb_path}: {e}")

if __name__ == "__main__":
    update_notebooks()
