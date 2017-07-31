# QM_2017_SSS_Team5
[![Build Status](https://travis-ci.org/MolSSI-SSS/QM_2017_SSS_Team5.svg?branch=master)](https://travis-ci.org/MolSSI-SSS/QM_2017_SSS_Team5)
[![codecov](https://codecov.io/gh/MolSSI-SSS/QM_2017_SSS_Team5/branch/master/graph/badge.svg)](https://codecov.io/gh/MolSSI-SSS/QM_2017_SSS_Team5)

We are team QM5 and we are 
* cici
* Sahil Gulania
* Charitha
* Kee (Team Leader)

# Objective

This is a project that consists of many sub-projects. All our project are wrapped into a Python module called `project`.

# Useful codes/notes

## Links

Check [MolSSI](https://github.com/MolSSI-SSS/Logistics_SSS_2017/blob/master/Links.md) for useful links.

## Github workflow

See notes for details. Here are some comments.


* In first establishment, there is a work flow as: upstream -(`fork`)-> origin -(`git clone`)-> local machine.

* After that, in order to keep everything in sync:

### upstream -/-> origin

* This direction works only for first time when `fork`.

### upstream --> local

* To check: `git remote -v`
* First time: `git remote add upstream [HTTP URL]`, then `git fetch upstream`
* Then `git reset --hard upstream/origin`: This deletes your local and duplicate upstream to your local.

### origin --> upstream

* `pull request` in web end.

### origin --> local

* First time: `git clone [HTTP URL]`
* After: use `git pull origin/master`

### local -/-> upstream

* Only person that has permission can push local to upstream. In addition, this operation is not safe. So consider this way forbidden.

### local --> origin:

`git add yourfile`
`git commit -m 'comment'`
`git push origin/master`
