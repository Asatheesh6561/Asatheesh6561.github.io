#!/bin/sh

# Build if needed
# jekyll build

git add _site
git commit -m "Update site"
git subtree push --prefix _site origin gh-pages
