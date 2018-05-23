from bs4 import BeautifulSoup

IGNORED_TAGS = ['del', 'strike', 's']
CODE_TAG = 'code'


def process_code(code):
    return 'CODELEXEM'


def get_child_text(node):
    name = getattr(node, 'name', None)
    if name in IGNORED_TAGS:
        return ''
    if name == CODE_TAG:
        return process_code(get_node_text(node))
    return get_node_text(node)


def get_node_text(node):
    if 'childGenerator' in dir(node):
        return ' '.join([get_child_text(child) for child in node.childGenerator()])
    return '' if node.isspace() else node


def process_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return ' '.join(list(filter(None, [get_node_text(node).strip() for node in soup.childGenerator()])))
