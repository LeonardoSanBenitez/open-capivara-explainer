from libs.utils.json_resilient import *


def test_split_string_except_inside_brackets():
    assert split_string_except_inside_brackets('{"a": "\n1"}\n{"a": "\n1"}') == ['{"a": "\n1"}', '{"a": "\n1"}']
    assert split_string_except_inside_brackets('das') == ['das']
    assert split_string_except_inside_brackets('') == []
    assert split_string_except_inside_brackets('''{"b": 2, "a": {}}''') == ['{"b": 2, "a": {}}']
    assert split_string_except_inside_brackets('{"name": "John", "age": Null}\n{"name": "Jane", "age": 25}\n{"name": ["Doe", "Mueller"], "age": 22}') == [  
        '{"name": "John", "age": Null}',  
        '{"name": "Jane", "age": 25}',  
        '{"name": ["Doe", "Mueller"], "age": 22}'  
    ]  
    assert split_string_except_inside_brackets('{"coords": [12,34], "active": true}\n{"coords": [56,78], "active": false},Extra text\n{"info": "some\\ninfo"}') == [  
        '{"coords": [12,34], "active": true}',  
        '{"coords": [56,78], "active": false}',  
        'Extra text',  
        '{"info": "some\\ninfo"}'  
    ]

def test_json_loads_resilient():
    worked, parsed = json_loads_resilient('{"a": 1, "b": 2}')
    assert worked
    
    worked, parsed = json_loads_resilient('{"a": 1, "b": 2}')
    assert worked
    
    worked, parsed = json_loads_resilient('{"a": "\n1"}')
    assert worked
    
    worked, parsed = json_loads_resilient('{"a": "\n1"}\n')
    assert worked
    
    worked, parsed = json_loads_resilient('{"a": "\n1"}\n')
    assert worked
    
    worked, parsed = json_loads_resilient(r'{"a": "\n1"}')
    assert worked
    
    worked, parsed = json_loads_resilient('{"a": "\n1"}\n{"a": "\n1"}')
    assert worked, len(parsed) == 2
    
    worked, parsed = json_loads_resilient('{"a": "\n1"}\n[1, 2, 3]\\n{"a": "\n1"}')
    assert worked, len(parsed) == 3