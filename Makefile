.PHONY: release-patch release-minor release-major release-patch-push release-minor-push release-major-push release-next build

release-next:
	@python scripts/release_tag.py patch --dry-run --allow-dirty

release-patch:
	@python scripts/release_tag.py patch

release-minor:
	@python scripts/release_tag.py minor

release-major:
	@python scripts/release_tag.py major

release-patch-push:
	@python scripts/release_tag.py patch --push

release-minor-push:
	@python scripts/release_tag.py minor --push

release-major-push:
	@python scripts/release_tag.py major --push

build:
	@python -m build
