# Git

## Create repo on Github from existing directory

1. Create repo on Github without `README`

2. Init
```
git init
```

2.bis. Define a `.gitignore` to exclude files from versioning
```
file/pattern/to/exclude
```

3. Add all files
```
git add .
```

4. Remove files
```
git rm --cached file
```

## Committing

```
git add file_containing_changes_to_be_committed
git commit file_containing_changes_to_be_committed -m "Commit message for this specific file"
```

To commit every changes
```
git commit -a -m "Commit message"
```

Pushing to Github
```
git push origin master
```