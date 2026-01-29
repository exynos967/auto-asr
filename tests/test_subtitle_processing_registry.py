from auto_asr.subtitle_processing.base import get_processor


def test_unknown_processor_raises():
    try:
        get_processor("nope")
        assert False
    except KeyError:
        assert True

