import functools
import os
import warnings

from dotenv import load_dotenv

from library.utils import PickleCache, NumpyCache, JSONCache


class CacheCenter:

    def __init__(self, root_path):
        self.root_path = root_path

    def to_file(self, *path, cacher):
        if self.root_path is not None and all(path):
            return cacher.tofile(os.path.join(self.root_path, *path))
        else:
            return self._null_decorator

    to_npz = functools.partialmethod(to_file, cacher=NumpyCache)
    to_pkl = functools.partialmethod(to_file, cacher=PickleCache)
    to_json = functools.partialmethod(to_file, cacher=JSONCache)

    @staticmethod
    def _null_decorator(func):
        return func


load_dotenv('.env')
cache_root_dir = os.getenv('DISK_CACHE_DIR')
if cache_root_dir is None:
    warnings.warn(
        "`cache_root_dir` is not given. The results of preprocessing won't be saved",
    )

cache_center = CacheCenter(cache_root_dir)
