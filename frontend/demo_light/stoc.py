"""https://github.com/arnaudmiribel/stoc"""

import re

import streamlit as st
import unidecode

DISABLE_LINK_CSS = """
<style>
a.toc {
    color: inherit;
    text-decoration: none; /* no underline */
}
</style>"""


def slugify(s: str) -> str:
    """
    Generate a URL-friendly slug from a string.
    Handles non-ASCII characters by transliterating them.
    e.g., "你好 世界" -> "ni-hao-shi-jie"
    """
    # Transliterate Unicode characters to ASCII
    s = unidecode.unidecode(s)
    # Convert to lowercase
    s = s.lower()
    # Replace non-alphanumeric characters with a hyphen
    s = re.sub(r'[^a-z0-9]+', '-', s)
    # Remove leading/trailing hyphens
    s = s.strip('-')
    return s


class stoc:
    def __init__(self):
        self.toc_items = list()

    def h1(self, text: str, write: bool = True):
        if write:
            st.title(text, anchor=slugify(text))
        self.toc_items.append(("h1", text))

    def h2(self, text: str, write: bool = True):
        if write:
            st.header(text, anchor=slugify(text))
        self.toc_items.append(("h2", text))

    def h3(self, text: str, write: bool = True):
        if write:
            st.subheader(text, anchor=slugify(text))
        self.toc_items.append(("h3", text))

    def toc(self, expander):
        st.write(DISABLE_LINK_CSS, unsafe_allow_html=True)
        # st.sidebar.caption("Table of contents")
        if expander is None:
            expander = st.sidebar.expander("**Table of contents**", expanded=True)
        with expander:
            with st.container(height=600, border=False):
                markdown_toc = ""
                for title_size, title in self.toc_items:
                    h = int(title_size.replace("h", ""))
                    markdown_toc += (
                        " " * 2 * h
                        + "- "
                        + f'<a href="#{slugify(title)}" class="toc"> {title}</a> \n'
                    )
                # st.sidebar.write(markdown_toc, unsafe_allow_html=True)
                st.write(markdown_toc, unsafe_allow_html=True)

    @classmethod
    def from_markdown(cls, text: str, expander=None):
        self = cls()
        # customize markdown font size
        # Generalize selectors to apply to st.title/header/subheader as well
        custom_css = """
        <style>
            /* Use more specific selectors to override Streamlit's default styles */
            [data-testid="stHeading"] h1, div[data-testid="stMarkdown"] h1 { font-family: 'Source Han Sans CN', sans-serif; font-size: 26px; color: #ec134d; }
            [data-testid="stHeading"] h2, div[data-testid="stMarkdown"] h2 { font-family: 'Source Han Sans CN', sans-serif; font-size: 23px; color: #f63366; }
            [data-testid="stHeading"] h3, div[data-testid="stMarkdown"] h3 { font-family: 'Source Han Sans CN', sans-serif; font-size: 20px; color: #f94877; }
            
            /* The following are for completeness, though h4/h5 are not used by the custom renderer yet */
            [data-testid="stHeading"] h4, div[data-testid="stMarkdown"] h4 { font-family: 'Source Han Sans CN', sans-serif; font-size: 19px; color: #fc5f89; }
            [data-testid="stHeading"] h5, div[data-testid="stMarkdown"] h5 { font-family: 'Source Han Sans CN', sans-serif; font-size: 18px; color: #fc5f89; }

            /* Adjust the font size for normal text within markdown */
            div[data-testid="stMarkdown"] p { font-family: 'Source Han Sans CN', sans-serif; font-weight: 200; font-size: 18px; }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

        non_header_lines = []
        def render_paragraph():
            if non_header_lines:
                st.markdown("\n".join(non_header_lines), unsafe_allow_html=True)
                non_header_lines.clear()

        for line in text.splitlines():
            if line.startswith("###"):
                render_paragraph()
                self.h3(line[3:].strip(), write=True)
            elif line.startswith("##"):
                render_paragraph()
                self.h2(line[2:].strip(), write=True)
            elif line.startswith("#"):
                render_paragraph()
                self.h1(line[1:].strip(), write=True)
            else:
                non_header_lines.append(line)
        
        render_paragraph() # Render any remaining content

        self.toc(expander=expander)
