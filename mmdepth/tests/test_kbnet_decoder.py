import sys
sys.path.append('.')
from mmdepth.models.decoders import KBNetDecoder
from mmengine import DefaultScope
default_scope = DefaultScope.get_instance(  # type: ignore
                'test',
                scope_name='mmdepth')
model = KBNetDecoder()
model.show_network()