"""
Sphinx extension to skip all private members starting with '_'
for autodoc, including inherited members.
"""

import inspect

KEYS_TO_SKIP = [
    # liquid audio
    "Mapping",
    "ChatState",
    # pydantic related
    "pydantic",
    "BaseModel",
    "model_fields",
    "model_extra",
    "model_computed_fields",
]


def custom_skip(app, what, name, obj, skip, options):
    """
    Hook for autodoc-skip-member.

    Parameters:
        app: Sphinx application
        what: type of the object ('module', 'class', 'exception', 'function', 'method', 'attribute')
        name: name of the member
        obj: the object itself
        skip: True/False if autodoc wants to skip it by default
        options: autodoc options
    Returns:
        True to skip, False to include, or None to fallback to default
    """
    if name.startswith("_"):
        return True  # skip private members

    for key in KEYS_TO_SKIP:
        if key in name:
            return True  # skip members containing specific keys
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        for key in KEYS_TO_SKIP:
            if key in obj.__qualname__:
                return True  # skip functions/methods of specific classes
    return None  # fallback to default


def setup(app):
    print("Setting up custom_skip extension...")
    app.connect("autodoc-skip-member", custom_skip)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
