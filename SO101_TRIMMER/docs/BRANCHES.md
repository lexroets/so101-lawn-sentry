# Two projects, one repository, three branches

`main` holds everything. `lawn-sentry` and `trimmer` are the same tree with the
other project's folder removed, so each can be cloned and run on its own
without carrying code it does not use.

```
main            core/  lawn_sentry/  trimmer/  training/  tools/  archive/
lawn-sentry     core/  lawn_sentry/            training/  tools/  archive/
trimmer         core/                trimmer/             tools/
```

## The thing to know before you start

`core/` is shared. A fix to the arm driver or the camera session belongs on
`main`, and the project branches get it by merging `main` into them:

```bash
git checkout lawn-sentry
git merge main
```

Work the other way — a change made on a project branch — has to be merged back
to `main` before the other branch can see it. That is the cost of splitting by
branch rather than by folder, and it is worth being deliberate about: day-to-day
work belongs on `main`, and the project branches are refreshed from it when
someone needs a clean single-project checkout.

## Refreshing a project branch after work on main

```bash
git checkout lawn-sentry
git merge main -m "sync with main"
git rm -r --cached trimmer 2>/dev/null; rm -rf trimmer
git commit -am "lawn-sentry: drop trimmer again"
git push origin lawn-sentry
```

The `rm -rf` is needed because merging `main` brings the other project's folder
back. If that feels fiddly, delete the branch and recreate it from `main` — it
is three commands and there is no history worth preserving on it.

## Why not separate repositories?

The two projects share the SO-101 driver, the servo configuration, the OAK-D
pipeline and the ground-plane maths. Two repositories would mean two copies of
`trimmer_arm.py` drifting apart, and the arm is the one piece of this system
where a silent difference between copies is genuinely dangerous.
