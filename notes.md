# Useful codes/notes

## Github workflow

See notes for details. Here are some comments.

* In first establishment, there is a work flow as: upstream -(fork)-> origin -(`git clone`)-> local machine.

* After that, in order to keep everything in sync:

### upstream -/-> origin

* This direction works only for first time when `fork`.

### origin --> upstream

* `pull request` in web end.

### upstream --> local

* To check: `git remote -v`
* First time: `git remote add upstream [HTTP URL]`, then `git fetch upstream`
* Then `git reset --hard upstream/origin`: This deletes your local and duplicate upstream to your local.

### local -/-> upstream

* Only person that has permission can push local to upstream. In addition, this operation is not safe. So consider this way forbidden.

### origin --> local

* First time: `git clone [HTTP URL]`
* After: use `git pull origin/master`

### local --> origin:

`git add yourfile`
`git commit -m 'comment'`
`git push origin/master`