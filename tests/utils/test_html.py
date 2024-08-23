from libs.utils.html import *


def test_is_valid_html():
    valid_html = [
        '<html><head><title>Example</title></head><body><h1>List Example</h1><ul><li>Item 1</li><li>Item 2</li><li>Item 3<ul><li>Nested Item 1</li><li>Nested Item 2</li></ul></li></ul></body></html>',
        '<!DOCTYPE html><html><head><title>Form Example</title></head><body><h1>Form Example</h1><form action="/submit" method="post"><label for="username">Username:</label><input type="text" id="username" name="username"><br><br><input type="submit" value="Submit"></form></body></html>',
        '<!DOCTYPE html><html><head><title>Simple Page</title></head><body><h1>Welcome to my page!</h1><p>This is a simple HTML page.</p></body></html>',
        '<!DOCTYPE html><html><head><title>Image Example</title></head><body><h1>Image Example</h1><img src="https://example.com/image.jpg" alt="Example Image"></body></html>',
        '<!DOCTYPE html><html><head><title>Link Example</title></head><body><h1>Link Example</h1><a href="https://example.com">Visit Example Website</a></body></html>',
        '<html><head><title>Test</title></head><body><h1>Hello, World!</h1></body></html>',
        'test',
        'test <strong>test</strong> test',
        '<ul>not broken</ul>',
        '<ul>broken</ul><ul>broken</ul>',
    ]

    invalid_html = [
        'broken</ul>',
        '<ul>broken',
        '<ul>broken</ul><ul>broken',
        '<ul>broken</il>',
    ]
    for i in range(len(valid_html)):
        html_string = valid_html[i]
        result, reason = is_valid_html(html_string)
        assert result, f'The string {i} {html_string[:10]}... is not recognized as valid HTML. Reason: {reason}'
    for html_string in invalid_html:
        result, reason = is_valid_html(html_string)
        assert not result, f'The string {html_string[:10]}... is not recognized as invalid HTML. Reason: {reason}'

def test_convert_markdown_to_html():
    assert convert_markdown_to_html('') == ''
    assert convert_markdown_to_html('hello') == 'hello'
    assert convert_markdown_to_html(' . hello ') == ' . hello '
    assert convert_markdown_to_html('```python\nprint("hello")\n```') == '<pre><code class="language-python"> print("hello")\n</code></pre>'
    assert convert_markdown_to_html('```PYTHON\nprint("hello")\n```') == '<pre><code class="language-python"> print("hello")\n</code></pre>'
    assert convert_markdown_to_html('```PYthOn\nprint("hello")\n```') == '<pre><code class="language-python"> print("hello")\n</code></pre>'
    assert convert_markdown_to_html('```python print("hello")\n```') == '<pre><code class="language-python"> print("hello")\n</code></pre>'
    assert convert_markdown_to_html('''```python\nprint("hello")\nprint('world')```''') == '''<pre><code class="language-python"> print("hello")\nprint('world')</code></pre>'''
    assert convert_markdown_to_html('<h1>Hello World</h1>') == '<h1>Hello World</h1>'

    # will be changed in the future
    assert convert_markdown_to_html('hello `inline` world') == 'hello `inline` world'
    assert convert_markdown_to_html('```java\nprint("hello")\n```') == '```java\nprint("hello")\n```'

    # out of scope, for now:
    # assert convert_markdown_to_html('```python\nprint("hello")\n```\n```python\nprint("world")\n```') == '<pre><code class="language-python">print("hello")\n</code></pre>\n<pre><code class="language-python">print("world")\n</code></pre>'
    # assert convert_markdown_to_html('```java\npublic class Main {\n    public static void main(String[] args) {\n        System.out.println("Hello World");\n    }\n}\n```') == '<pre><code class="language-java">public class Main {\n    public static void main(String[] args) {\n        System.out.println("Hello World");\n    }\n}\n</code></pre>'

def test_strip_graph_from_answer():
    # code
    text, code = strip_graph_from_answer('batata')
    assert text == 'batata'
    assert code == ''

    text, code = strip_graph_from_answer('batata <doce>')
    assert text == 'batata <doce>'
    assert code == ''

    text, code = strip_graph_from_answer('batata <code claaaa')
    assert text == 'batata <code claaaa'
    assert code == ''

    text, code = strip_graph_from_answer('')
    assert text == ''
    assert code == ''

    text, code = strip_graph_from_answer('\n')
    assert text == ''
    assert code == ''

    # markdown
    text, code = strip_graph_from_answer('batata ```python-viz\nimport matplotlib.pyplot as plt\n```')
    assert text == 'batata'
    assert code == 'import matplotlib.pyplot as plt'

    text, code = strip_graph_from_answer('batata\nmandioca\n```python-viz\nimport matplotlib.pyplot as plt``` das')
    assert text == 'batata\nmandioca'
    assert code == 'import matplotlib.pyplot as plt'

    text, code = strip_graph_from_answer('batata\n```python-viz\nimport matplotlib.pyplot as plt')
    assert text == 'batata'
    assert code == 'import matplotlib.pyplot as plt'


    text, code = strip_graph_from_answer('  \n  batata\n```python-viz```  \ndasdas')
    assert text == 'batata'
    assert code == ''

    text, code = strip_graph_from_answer('''
    <p>batata</p>
    <table>
      <tr>
        <th>Month</th>
        <th>Sales</th>
      </tr>
      <tr>
        <td>January</td>
        <td>100</td>
      </tr>
    </table>
    ```python-viz
    print('hello world')                                     
    ```
    ''')
    assert len(text) == 176
    assert code.strip() == 'print(\'hello world\')'

    # html
    # all test cases are basically the same as the markdown ones, but with html tags
    text, code = strip_graph_from_answer('batata <code class="language-python-viz">import matplotlib.pyplot as plt</code>')
    assert text == 'batata'
    assert code == 'import matplotlib.pyplot as plt'

    text, code = strip_graph_from_answer('batata\nmandioca\n<code class="language-python-viz">import matplotlib.pyplot as plt</code> das')
    assert text == 'batata\nmandioca'
    assert code == 'import matplotlib.pyplot as plt'

    text, code = strip_graph_from_answer('batata\n<code class="language-python-viz">import matplotlib.pyplot as plt')
    assert text == 'batata'
    assert code == 'import matplotlib.pyplot as plt'

    text, code = strip_graph_from_answer('  \n  batata\n<code class="language-python-viz"></code>  \ndasdas')
    assert text == 'batata'
    assert code == ''

    text, code = strip_graph_from_answer('''
    <p>batata</p>
    <table>
      <tr>
        <th>Month</th>
        <th>Sales</th>
      </tr>
      <tr>
        <td>January</td>
        <td>100</td>
      </tr>
    </table>
    <code class="language-python-viz">
    print('hello world')                                     
    </code> dasdas        
    ''')
    assert len(text) == 176
    assert code.strip() == 'print(\'hello world\')'

def compose_html_from_code():
    # empty
    status_code, html_viz = compose_html_from_code('''print("hello")''', verbose = False, viz_notebook = False)
    assert status_code // 100 == 2
    assert html_viz == ''
    
    status_code, html_viz = compose_html_from_code('''import matplotlib.pyplot''', verbose = False, viz_notebook = False)
    assert status_code // 100 == 2
    assert html_viz == ''
    
    status_code, html_viz = compose_html_from_code('  ', verbose = False, viz_notebook = False)
    assert status_code // 100 == 2
    assert html_viz == ''
    
    status_code, html_viz = compose_html_from_code('', verbose = False, viz_notebook = False)
    assert status_code // 100 == 2
    assert html_viz == ''
    
    # viz
    status_code, html_viz = compose_html_from_code('''import matplotlib.pyplot as plt; plt.plot([1, 2, 3, 4, 5], [2, 3, 5, 7, 11]); plt.show()''', verbose = True, viz_notebook = False)
    assert status_code // 100 == 2
    assert len(html_viz) > 0
    
    status_code, html_viz = compose_html_from_code('''import matplotlib.pyplot as plt; plt.scatter([1], [2])''', verbose = True, viz_notebook = False)
    assert status_code // 100 == 2
    assert len(html_viz) > 0
    
    status_code, html_viz = compose_html_from_code('''import matplotlib.pyplot as plt; plt.bar(['A'], [2]);''', verbose = True, viz_notebook = False)
    assert status_code // 100 == 2
    assert len(html_viz) > 0
    
    # invalid
    status_code, html_viz = compose_html_from_code('''print('hello''', verbose = False, viz_notebook = False)
    assert status_code // 100 == 4
    assert html_viz == ''