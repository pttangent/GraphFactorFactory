from graphfactorfactory.domain.config import BuildConfig


def test_qlib_cache_is_off_by_default():
    assert BuildConfig().store_qlib_cache is False
