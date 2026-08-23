from src.visual.preprocess import list_images


def test_list_images_accepts_empty_reference_folder(tmp_path):
    assert list_images(tmp_path) == []
