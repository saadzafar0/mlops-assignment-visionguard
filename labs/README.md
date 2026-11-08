# Tool labs

These exercises are separate from the real VisionGuard training and serving code.

## DVC CSV exercise (Task 1.4)

Run `python labs/dvc_toy.py` from the project root in the activated environment.
It refuses to reuse its output folder so an earlier exercise is not overwritten.
The nested Git repository and its local remote live under ignored `labs/work/`.

The completed exercise created two commits:

| Version | Commit | CSV change |
| --- | --- | --- |
| First | `851b3c4` | One real-image row |
| Second | `5a5fe62` | Add an AI-image row |

Both versions were pushed to a local DVC remote. Checking out the first Git commit
and running `dvc checkout` recovered the first CSV. After deleting that CSV and the
toy's DVC cache, `dvc pull` fetched it again from remote storage. Returning to
`main` and pulling recovered the second CSV.

The first implementation tried to use `dvc checkout` when returning to `main`
after clearing the cache. That failed because the second CSV version was no longer
cached. The final script correctly uses `dvc pull` at that point. The first attempt
is retained under ignored `labs/work/dvc-toy-first-attempt/`.

Evidence is in `reports/generated/dvc_toy/transcript.txt` and `commits.txt`.
Capture terminal screenshots of the commits and restoration steps for the report.
The toy commits have **not** been published to DagsHub; that requirement remains.

## Definitions

- **DVC pointer:** a small tracked description of data, including its path and
  checksum, which links a Git revision to bytes stored outside Git.
- **Cache:** local content-addressed copies of data that DVC can use to reconstruct
  a working dataset without downloading the same bytes again.
- **Remote:** storage used to share or recover cached data across machines.
- **`dvc checkout`:** materialises the version described by the current pointer
  from the local cache.
- **`dvc push` / `dvc pull`:** transfer cached objects to/from remote storage.
