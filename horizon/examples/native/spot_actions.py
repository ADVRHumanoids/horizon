"""Supported native Spot actions and their command-line entry point."""

ACTIONS = ("wheelie", "jump_up", "jump_forward", "jump_on_wall", "leap", "jump_twist")


def main():
    from horizon.examples.native.spot_motions import main as run_motions

    return run_motions()


if __name__ == "__main__":
    raise SystemExit(main())