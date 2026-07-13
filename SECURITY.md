# Security policy

Report security issues privately to the repository maintainers rather than in a public issue.

Checkpoint files may contain pickle data and can execute arbitrary code when loaded. Only load checkpoints produced by a trusted local run. Inference should use the weights-only `safetensors` files produced by this package. No model file may be committed, uploaded as an Actions artifact, or attached to a GitHub release.

Do not include patient data, protected health information, credentials, DICOM headers, or institutional filesystem paths in bug reports.

The public repository must not contain patient-derived data, medical images, DICOM/NIfTI files, cohort tables, trained weights, checkpoints, predictions, metrics, or experiment outputs. Only explicitly synthetic CSV examples under `examples/` are permitted.

Generated DICOM files may retain patient or study metadata from their local source series. This project does not claim to provide a validated DICOM de-identification pipeline. Use an institutionally approved de-identification workflow before sharing any generated DICOM.
