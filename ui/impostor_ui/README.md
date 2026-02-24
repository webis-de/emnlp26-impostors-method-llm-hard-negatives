# Impostor UI

Single-page Tkinter UI for running the impostor detector with:
- Original or ablation methods
- Dynamic parameter inputs per method
- Impostor generation technique selection
- Disputed/candidate input by file upload or pasted text
- Start button, loading state, progress bar, results, and next-test reset

## Run

From repository root:

```bash
python3 ui/impostor_ui/app.py
```

## Notes

- The UI stores ad-hoc text inputs in MongoDB so the detector can resolve IDs.
- Some generation modes depend on configured external APIs and can take a long time.
- Tkinter is part of the Python standard library (no Streamlit required).
