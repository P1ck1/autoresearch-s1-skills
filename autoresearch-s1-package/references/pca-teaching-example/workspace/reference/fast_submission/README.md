`pca` here is a **self-contained copy of `reference/fast_pca.py`** (with a `#!/usr/bin/env python3`
shebang) so it is a legitimate standalone submission that passes the pure-numpy/scipy allowlist
(it does not `import reference`). Regenerate after editing the reference:

    { echo '#!/usr/bin/env python3'; cat reference/fast_pca.py; } > reference/fast_submission/pca
    chmod +x reference/fast_submission/pca
