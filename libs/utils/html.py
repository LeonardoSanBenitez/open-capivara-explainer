from lxml import html
from lxml.etree import LxmlError
from typing import Tuple, Dict, Optional
import traceback
from contextlib import redirect_stdout
import io
import base64
import re
import matplotlib.pyplot as plt
import logging

from libs.utils.logger import get_logger

logger = get_logger(f'libs.helper.html', level = logging.WARNING)


def is_valid_html(html_string) -> Tuple[bool, str]:
    '''
    Performs a semi-strict check to see if a string is valid HTML.
    TODO: this should not be that difficult. Maybe using a library like html5lib is easier
    @return bool worked: if the HTML is valid
    @return str error: the error message if the HTML is not valid
    '''
    try:
        # Check if the parsed HTML has a single root element
        '''
        parsed_html = html.fromstring(html_string)
        if parsed_html.tag not in ('html', 'body'):
            # If the root tag is not html or body, it's a fragment, thus invalid
            return False
        '''

        # Check if serialized HTML is wrapped with html and body tags
        '''
        if not serialized_html.strip().startswith('<html') or not serialized_html.strip().endswith('</html>'):
            return False
        '''

        # Check if serialization and re-parsing don't change the structure
        tree = html.fromstring(html_string)
        if html.tostring(tree) != html.tostring(html.fromstring(html.tostring(tree))):
            return False, 'the HTML structure is not preserved after serialization and re-parsing'

        # Check for additional parsing errors
        parser = html.HTMLParser()
        html.document_fromstring(html_string, parser=parser)
        if parser.error_log:
            return False, str(parser.error_log)

        html_string = html_string.replace('> ', '>')
        html_string = html_string.replace('< ', '<')

        # Check for unclosed gt and lt
        if html_string.count('<') != html_string.count('>'):
            return False, 'the HTML contains unclosed < or >'

        # Check unmatche for some tags
        mandatory_match_tags = ['ul', 'ol', 'strong', 'p', 'em']
        for tag in mandatory_match_tags:
            if html_string.count(f'<{tag}>') != html_string.count(f'</{tag}>'):
                return False, f'the HTML contains matches <{tag}> or </{tag}>'

        return True, ''
    except LxmlError:
        return False, 'the HTML is not well-formed'
    except ValueError:
        return False, 'the document is empty'


def convert_markdown_to_html(markdown: str) -> str:
    '''
    Really basic function, not a general converter.

    Implemented prioritizing being reliable (not breaking the text) over being complete.

    Does not raise exceptions, but logs them.
    '''
    try:
        html = markdown

        # python block appears exactly once
        if (len(re.findall(r'```', html)) == 2) and (len(re.findall(r'(?i)```python( |\n)', html)) == 1):
            html = re.sub(r'(?i)```python( |\n)', '<pre><code class="language-python"> ', html)
            html = re.sub(r'```', '</code></pre>', html)

        # print(html)
        worked, error = is_valid_html(html)
        if (not worked) and (markdown != ''):
            raise ValueError(f"Invalid HTML was accidentaly generated, error: {error}. Generated HTML: {html}")
        return html
    except Exception as e:
        print(666)
        logger.warning(f"Unexpected Error at convert_markdown_to_html: {e}, {traceback.format_exc()}. Returning original markdown.")
        return markdown


def strip_graph_from_answer(answer: str) -> Tuple[str, str]:
    '''
    Can handle both markdown and html
    Does not handle both mixed.
    Does not handle multiple code cells.
    Should never raise any exception.

    :param answer: the answer to be processed
    :return text: natural language explanation
    :return code: the code that generates the visualization
    '''
    text = answer
    code = ''

    # Extract code - Markdown
    splits = answer.split("```python-viz")
    if len(splits) > 1:
        code = answer.split("```python-viz")[1]
        code = code.split("```")[0]
        text = splits[0]
        if len(splits) > 2:
            logging.warning('strip_graph_from_answer: more than one code cell passed')

    # Extract code - HTML
    if (code == '') and (text.strip() != ''):
        tree = html.fromstring(answer)
        code_cells = tree.xpath('//code[@class="language-python-viz"]')
        if len(code_cells) > 0:
            code = code_cells[0].text or ''
            text = answer.split('<code')[0]
            if len(code_cells) > 1:
                logging.warning('strip_graph_from_answer: more than one code cell passed')

    # Post process text
    text = text.strip()

    # Post process code
    code = code.replace('```', '')
    code = code.strip()

    return text, code


def compose_html_from_code(code: str, alt: str = 'Graph visualizing the data', viz_notebook: bool = True, verbose: bool = True) -> Tuple[int, str]:
    '''
    Returns a valid HTML, even if code is invalid.
    Empty string is considered valid HTML.

    :param code: the code to be executed
    :param alt: the alt text for the image
    :param viz_notebook: if True, will display the image in the notebook. May not work if the caller is not a IPython Notebook.
    '''
    if len(code) == 0:
        return 204, ''

    buffer = io.BytesIO()
    plt.close('all')
    plt.switch_backend('Agg')
    plt.close('all')
    try:
        # Redirect is used to supress the outputs/prints
        with redirect_stdout(io.StringIO()):
            exec(code, {}, {})
    except Exception as e:
        if verbose:
            logging.error(f'compose_html_from_code: error executing code: {e}, {traceback.format_exc()}')
        return 422, ''
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    # If code is valid, not empty, but does not generate an image, return 204 and ''
    # plt generates an empty white square
    # These thresholds are empirical
    if len(plt.gca().get_children()) <= 10:
        return 204, ''
    if buffer.tell() <= 2396:
        return 204, ''
    if len(plt.gcf().axes) == 0:
        return 204, ''

    # vizualize
    if viz_notebook:
        image_data = base64.b64decode(image_base64)
        from IPython.display import display, Image
        display(Image(image_data))
    logging.info(f'compose_html_from_code: code executed successfully, returning image with {len(image_base64)} bytes.')
    return 200, f'<img src="data:image/png;base64,{image_base64}" alt="{alt}">'


def strip_graph_html_from_answer(answer: str, add_image_placeholder: bool = False, placeholder_text: str = '\n[A graph was generated]') -> Tuple[str, Optional[str]]:
    """
    Extracts the HTML and base64 image from the answer of a flow.

    For the moment, only takes the text before the first tag
    """
    text = answer.split('<img')[0]

    try:
        img_source = re.search(r'img src="(.+?)">', answer, re.DOTALL).group(1)  # type: ignore
        image_base64 = re.search(r'data:image/png;base64,(.+?)"', img_source).group(1)  # type: ignore
        assert len(image_base64) > 0
        assert image_base64 != 'IMAGE_DATA'
        # print('>>>>>>>> RECEIVED IMAGE: ', image_base64[:30] + '...')
    except:  # noqa
        image_base64 = None

    if image_base64 and add_image_placeholder:
        text += placeholder_text
    return text, image_base64


def display_HTML_answer(answer: str) -> None:
    from IPython.display import display, Image, HTML
    text, image_base64 = strip_graph_html_from_answer(answer)
    display(HTML(text))
    if image_base64 is not None:
        display(Image(base64.b64decode(image_base64)))
    else:
        # display(HTML('No graph was generated'))
        pass
