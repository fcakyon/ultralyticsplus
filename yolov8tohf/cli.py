import fire
from .hf_utils import push_to_hfhub


def app() -> None:
    """Cli app."""
    fire.Fire(push_to_hfhub)


if __name__ == "__main__":
    app()
