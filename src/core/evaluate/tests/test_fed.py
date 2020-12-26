from unittest.mock import call, patch

from ..fed import FEDModel


def test_cache():
    with patch.object(FEDModel, 'build_graph') as mock_build_graph, \
        patch.object(FEDModel, 'encode'):
        a1 = FEDModel.download_from(hub_url='a')
        a2 = FEDModel.download_from(hub_url='a')
        b = FEDModel.download_from(hub_url='b')

    assert a1 is a2  # from same url
    assert a1 is not b
    mock_build_graph.assert_has_calls([call('a'), call('b')])
