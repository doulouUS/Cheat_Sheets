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

## Explore files

List all tracked files under `master` branch
```
git ls-tree -r master --name-only
```

## Branches

![](./images/git_branches.jpeg)

List all branches
```
git branch -a
OR
git show-branch  // branch and commit 
```

Create local branch
```
git branch my-new-branch
git checkout new-branch  // switch to that branch 
OR
git checkout -b my-new-branch
```

Create remote branch (work on a local branch)
```
git push remote-name branch-name  // typically remote-name is origin, branch-name is the same name for the local and remote branch
```


## Pull request

1. Fork
2. Commit changes, create new branch
3. Pull request on github's web interface

