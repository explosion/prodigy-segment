from prodigy_segment import get_base64_string


def test_get_base64_string():
    example = {"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAgAAZABkAAD"}
    assert get_base64_string(example['image']) == "/9j/4AAQSkZJRgABAgAAZABkAAD"
