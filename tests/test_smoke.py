import ivchar
from ivchar.cli import main


def test_version():
    assert ivchar.__version__ == "0.1.0"


def test_cli_runs(capsys):
    main([])
    assert "ivchar" in capsys.readouterr().out
