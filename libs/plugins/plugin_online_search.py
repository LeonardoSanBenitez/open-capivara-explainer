from typing import Any
from bs4 import BeautifulSoup
import re
import requests

from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel import KernelContext
from semantic_kernel.kernel_pydantic import KernelBaseModel


def decode_str(string):
    return string.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def get_page_sentence(page, count: int = 10):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # find all sentence
    sentences = []
    for p in paragraphs:
        sentences += p.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    # get first `count` number of sentences
    return ' '.join(sentences[:count])


def remove_nested_parentheses(string):
    pattern = r'\([^()]+\)'
    while re.search(pattern, string):
        string = re.sub(pattern, '', string)
    return string


def is_integer(s: Any) -> bool:
    # TODO: move this to utils
    try:
        int(s)
        return True
    except:  # noqa
        return False


class PluginOnlineSearch(KernelBaseModel):
    @kernel_function(
        name = "search_wikipedia",
        description = "Search this entity name on Wikipedia and returns the first sentences. If wikipedia does not have the entity, it will return some related entities to search next.",
    )
    @kernel_function_context_parameter(
        name = "entity",
        description = "Exact name of the entity to search on Wikipedia.",
        type = "string",
        required = True,
    )
    @kernel_function_context_parameter(
        name = "count",
        description = "Number of sentences to return. Default is 10.",
        type = "integer",
        required = False,
        default_value = 10,
    )
    def search_wikipedia(self, context: KernelContext) -> str:
        print(f"entered search_wikipedia: {context.variables}")
        # Parameter validation
        if 'entity' not in context.variables:
            return "Error: missing required parameter 'entity'."
        if type(context.variables['entity']) != str:
            return "Error: parameter 'entity' should be an string."
        entity: str = context.variables['entity']

        if 'count' in context.variables:
            if is_integer(context.variables['count']):
                return "Error: parameter 'count' should be an integer."
            count: int = int(context.variables['count'])
        else:
            count = 10

        # Exec
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35"
        }
        response_text = requests.get(search_url, headers=headers).text
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            result_titles = [decode_str(div.get_text().strip()) for div in result_divs]
            result_titles = [remove_nested_parentheses(result_title) for result_title in result_titles]
            obs = f"Could not find {entity}. Similar: {result_titles[:5]}."
        else:
            page_content = [p_ul.get_text().strip() for p_ul in soup.find_all("p") + soup.find_all("ul")]
            if any("may refer to:" in p for p in page_content):
                obs = self.search_wikipedia("[" + entity + "]")
            else:
                page = ""
                for content in page_content:
                    if len(content.split(" ")) > 2:
                        page += decode_str(content)
                    if not content.endswith("\n"):
                        page += "\n"
                obs = get_page_sentence(page, count=count)
        return obs
