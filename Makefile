predict:
	@sudo cog predict -i prompt=tree -i hires=True

push:
	@sudo cog push r8.im/elct9620/pastel-mix

setup:
	@sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
	@sudo chmod +x /usr/local/bin/cog
	@pip3 install -r scripts/requirements.txt

download: setup
	@python3 scripts/download.py

# Sync to remote GPU server
sync:
	rsync -av -e ssh --exclude='.git/' --exclude='**/.mypy_cache/' . $(server):~/cog
