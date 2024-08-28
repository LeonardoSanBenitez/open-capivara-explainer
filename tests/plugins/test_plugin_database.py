import pytest
from libs.plugins.plugin_database import _detect_embedding_call, _replace_embedding_call


def test_detect_embedding_call_basic():
    q = '''SELECT chunks, publisheddate 
    FROM items 
    ORDER BY embedding <-> embedded_question('renewable energies')'''
    topic = _detect_embedding_call(q)
    assert topic == 'renewable energies'

def test_detect_embedding_call_no_match():
    q = '''SELECT chunks, publisheddate 
    FROM items'''
    topic = _detect_embedding_call(q)
    assert topic is None

def test_detect_embedding_call_multiple_lines():
    q = '''SELECT chunks, publisheddate 
    FROM items 
    WHERE column1 = 'value'
    ORDER BY embedding <-> embedded_question(
        'climate change and renewable energy'
    )'''
    topic = _detect_embedding_call(q)
    assert topic == 'climate change and renewable energy'

def test_detect_embedding_call_spaces_inside():
    q = '''SELECT chunks, publisheddate 
    FROM items 
    ORDER BY embedding <-> embedded_question ( '   AI in healthcare  ' )'''
    topic = _detect_embedding_call(q)
    assert topic == '   AI in healthcare  '

def test_detect_embedding_call_nested_parentheses():
    q = '''SELECT chunks, publisheddate 
    FROM items 
    ORDER BY embedding <-> embedded_question('energy (renewable) sources')'''
    topic = _detect_embedding_call(q)
    assert topic == 'energy (renewable) sources'

def test_detect_embedding_call_no_parentheses():
    q = '''SELECT chunks, publisheddate 
    FROM items 
    ORDER BY embedding <-> embedded_question 'just a string' '''
    topic = _detect_embedding_call(q)
    assert topic is None

def test_detect_embedding_call_empty():
    q = '''SELECT chunks, publisheddate 
    FROM items 
    ORDER BY embedding <-> embedded_question('')'''
    topic = _detect_embedding_call(q)
    assert topic is None

def test_detect_embedding_call_multiple_embedded_questions():
    q = '''SELECT chunks, publisheddate 
    FROM items 
    ORDER BY embedding <-> embedded_question('first topic'), 
    other_column <-> embedded_question('second topic')'''
    
    with pytest.raises(Exception):
        _detect_embedding_call(q)

####

def test_replace_embedding_call_basic():
    assert _replace_embedding_call(
        """SELECT chunks, publisheddate 
        FROM items 
        ORDER BY embedding <-> embedded_question('renewable energies')""",
        [0.1, 0.2]
    ) == """SELECT chunks, publisheddate 
        FROM items 
        ORDER BY embedding <-> '[0.1, 0.2]'"""

def test_replace_embedding_call_no_match():
    with pytest.raises(RuntimeError):
        _replace_embedding_call(
            '''SELECT chunks, publisheddate 
            FROM items''',
            [0.1, 0.2]
        )

def test_replace_embedding_call_multiple_calls():
    with pytest.raises(RuntimeError):
        _replace_embedding_call(
            '''SELECT chunks, publisheddate 
            FROM items 
            ORDER BY embedding <-> embedded_question('first topic'), 
            other_column <-> embedded_question('second topic')''',
            [0.1, 0.2]
        )

def test_replace_embedding_call_nested_parentheses():
    assert _replace_embedding_call(
        """SELECT chunks, publisheddate 
        FROM items 
        ORDER BY embedding <-> embedded_question('energy (renewable) sources')""",
        [0.3, 0.4]
    ) == """SELECT chunks, publisheddate 
        FROM items 
        ORDER BY embedding <-> '[0.3, 0.4]'"""
