from libs.utils.parse_llm import match_key_from_candidates


def test_match_key_from_candidates():
    assert match_key_from_candidates('get_sec', ['getSec']) == 'getSec'
    assert match_key_from_candidates('batata', ['batata']) == 'batata'
    assert match_key_from_candidates('batata', ['batata', 'queijo']) == 'batata'
    assert match_key_from_candidates('HELLO', ['hello']) == 'hello'
    assert match_key_from_candidates('Hello', ['world', 'hello', 'Hello']) == 'Hello'
    assert match_key_from_candidates('world', ['Hello', 'WORLD', 'planet', 'earth']) == 'WORLD'
    assert match_key_from_candidates('python', ['java', 'javascript', 'ruby', 'scala'], required = False) is None
    assert match_key_from_candidates('DaTa_ScIeNcE', ['data_science', 'Data_Science', 'data', 'science'], required = False) is None
    assert match_key_from_candidates('construir', ['casa/construir', 'casa/destruir', 'casa/reformar']) == 'casa/construir'
    assert match_key_from_candidates('Construir', ['casa/construir', 'casa/destruir', 'casa/reformar']) == 'casa/construir'
    assert match_key_from_candidates('!@#$%^&*()', ['!@#$%^&*()', '1234567890', 'abcdefghijklmnopqrstuvwxyz']) == '!@#$%^&*()'
    assert match_key_from_candidates('batata doce', ['batata', 'cenoura'], custom_simplifications={'root': lambda x: x.replace('doce', '').strip()}) == 'batata'
    assert match_key_from_candidates('Batata', ['batata', 'queijo', 'QUEIJO']) == 'batata'
    assert match_key_from_candidates('Batata', ['batata', 'BATATA', 'QUEIJO'], required=False) == None
