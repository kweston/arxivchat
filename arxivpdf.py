import pdfplumber
from operator import itemgetter
from itertools import groupby
import fitz
import pandas as pd
import re


class ArxivPDF:

    def __init__(self, fname):
        self.fname = fname

    @staticmethod
    def _is_footnote(row: pd.Series) -> bool:
        return re.match(r'^\d+[a-zA-z]+', row.text.strip()) is not None

    @staticmethod
    def _is_section_header(row: pd.Series) -> bool:
        return re.match(r'^\d+\. ', row.text.strip()) is not None
        
    def get_text_dataframe(self) -> pd.DataFrame:


        with fitz.open(self.fname) as doc:
            # get the width and height of the first page
            dfs = []
            for i, page in enumerate(doc):
                width, height = page.rect[2:]
                pdata = page.get_text("blocks")
                df = pd.DataFrame(pdata, columns=['x0', 'y0', 'x1', 'y1', 'text', 'block_no', 'block_type'])
                # assume that text to the left of center are in the first column
                # assume that text to the right of center are in the second column
                # filter out blocks which cross columns
                df_left =  df.loc[(df.x0 < width // 2) & (df.x1 < width // 2)]
                df_right =  df.loc[(df.x0 > width // 2) & (df.x1 > width // 2)]
                df_combo = pd.concat([df_left, df_right])
                # filter out images
                df_combo = df_combo.loc[df_combo.block_type == 0]
                # filter out vertical text
                df_combo = df_combo.loc[df_combo.x1 - df_combo.x0 > 0.5 * (df_combo.y1 - df_combo.y0)]
                # filter out footnotes
                df_combo = df_combo.loc[~df_combo.apply(self._is_footnote, axis=1)]
                df_combo['page_no'] = i
                dfs.append(df_combo)

        return pd.concat(dfs)

    def get_text(self, split_sections=False) -> str:
        """ Extract text from a an arxiv pdf in 2 column format"""

        df = self.get_text_dataframe()
        sections = [""]
        section_names = []
        prev_page = -1
        for ind, row in df.iterrows():
            if split_sections:
                if self._is_section_header(row):
                    sections.append("")
                    section_names.append(row.text.strip())

            if row.page_no != prev_page:
                sections[-1] += row.text
            else:
                sections[-1] += row.text + "\n"
            prev_page = row.page_no
        
        if split_sections:
            return sections, section_names
        else:
            return sections[0]



if __name__ == "__main__":
    fname = "2302.00923v4_clean.pdf"
    pdf = ArxivPDF(fname)
    print(f"Extracting text from {fname}")
    text = pdf.get_text()
    outfname = fname.replace('pdf', 'txt')
    print(f"Writing to {outfname}")
    with open(outfname, 'w') as f:
        f.write(text)