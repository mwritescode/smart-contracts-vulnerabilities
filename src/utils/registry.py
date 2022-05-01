class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides register functions.
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    '''
    
    # Instanciated objects will be empyty dictionaries
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    # Decorator factory. Here self is a Registry dict
    def register(self, module_name, module=None):
        
        # Inner function used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # Inner function used as decorator -> takes a function as argument
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn
    
def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module

REGISTRY = Registry()