from music_transcription.loader.save_files import try_read_file
from unittest.mock import patch, mock_open
import pickle
import io


def should_not_be_called():
    raise RuntimeError()


def test_try_read_file__file_exists():
    """
    Try to read a file that exists
    we check that its content is returned
    """
    data = "hello there"
    file_content = pickle.dumps(data)
    with patch("builtins.open", mock_open(read_data=file_content)) as mock_file:
        return_value = try_read_file("save_path", should_not_be_called)
        mock_file.assert_called_with("save_path", "rb")
        assert return_value == data


def test_try_read_file__file_doesnt_exist():
    """
    Try to read a file that doesn't exist
    we check that the function is called and its content saved and returned
    """
    data = "this is the some data"
    fake_file = mock_open()
    with patch("builtins.open", fake_file) as mock_file:
        return_value = try_read_file("no/exist/path", lambda: data)
        assert return_value == data

        file_content = mock_file().write.call_args.args[0]
        assert pickle.load(io.BytesIO(file_content)) == data
